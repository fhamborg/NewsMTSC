import argparse
import json
from collections import defaultdict

from diskdict import DiskDict

import jsonlines
import pandas as pd

from NewsSentiment.fxlogger import get_logger

logger = get_logger()


def rename_flatten(dictionary, key_prefix):
    new_dict = {}

    for k, v in dictionary.items():
        new_k = key_prefix + "-" + k
        new_dict[new_k] = v

    return new_dict


def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}


def non_scalar_to_str(d):
    new_d = {}
    for k, v in d.items():
        if type(v) in [list, dict]:
            new_v = json.dumps(v)
        else:
            new_v = v
        new_d[k] = new_v
    return new_d


COL_NAME_TMP_GROUP = "_tmp_named_id_"


def _find_run_ids(completed_tasks: dict):
    """
    First check the maximum number of run ids, then ensures that each experiment has
    that many.
    :param completed_tasks:
    :return:
    """
    run_ids = set()
    for named_id, result in completed_tasks.items():
        vals = without_keys(result, ["details"])
        run_id = vals["run_id"]
        run_ids.add(run_id)
        del vals["run_id"]
        del vals["experiment_named_id"]
        del vals["experiment_id"]
        named_experiment_id_wo_run_id = json.dumps(vals)
        result[COL_NAME_TMP_GROUP] = named_experiment_id_wo_run_id
    num_runs_per_experiment = len(run_ids)
    logger.info("found %s run_ids: %s", num_runs_per_experiment, run_ids)

    # check that each experiment has as many run ids
    named_id2run_ids = defaultdict(list)
    for named_id, result in completed_tasks.items():
        vals = without_keys(result, ["details"])
        named_experiment_id_wo_run_id = result["_tmp_named_id_"]
        run_id = vals["run_id"]
        named_id2run_ids[named_experiment_id_wo_run_id].append(run_id)

    count_too_few_runs = 0
    for named_id, run_ids in named_id2run_ids.items():
        if len(run_ids) != num_runs_per_experiment:
            logger.debug("%s runs for %s", len(run_ids), named_id)
            count_too_few_runs += 1
    if count_too_few_runs == 0:
        logger.info(
            "GOOD: num experiments with too few runs: %s of %s",
            count_too_few_runs,
            len(named_id2run_ids),
        )
    else:
        logger.warning(
            "num experiments with too few runs: %s of %s",
            count_too_few_runs,
            len(named_id2run_ids),
        )

    return num_runs_per_experiment, completed_tasks


def _aggregate_and_mean(df: pd.DataFrame):
    df = df.copy(deep=True)
    col_names_original_order = list(df.columns)
    df_aggr = df.groupby(COL_NAME_TMP_GROUP).mean()

    # from https://stackoverflow.com/a/35401886
    # this creates a df that contains aggregated values (from df_aggr) and also
    # all other columns (non-aggregated)
    aggr_col_names = list(df_aggr.columns)
    df.drop(aggr_col_names, axis=1, inplace=True)
    df.drop_duplicates(subset=COL_NAME_TMP_GROUP, keep="last", inplace=True)
    df = df.merge(
        right=df_aggr, right_index=True, left_on=COL_NAME_TMP_GROUP, how="right"
    )

    # reorder the dataframe to have the established order of columsn
    # taken from: https://stackoverflow.com/a/13148611
    df = df[col_names_original_order]

    # delete temp col
    del df[COL_NAME_TMP_GROUP]

    return df


def _dfs_to_excel(pathname, name2df):
    writer = pd.ExcelWriter(pathname, engine="xlsxwriter")
    for name, df in name2df.items():
        if df is None:
            logger.info("skipping df because empty: %s", name)
            continue

        df.to_excel(writer, sheet_name=name, startrow=0, startcol=0)
    writer.save()


def shelve2xlsx(opt, ignore_graceful_exit_experiments):
    completed_tasks = DiskDict(opt.results_path)
    logger.info(
        "found {} results in file {}".format(len(completed_tasks), opt.results_path)
    )
    # get max run id
    num_runs_per_experiment, completed_tasks = _find_run_ids(completed_tasks)

    flattened_results = {}

    for named_id, result in completed_tasks.items():
        if result["rc"] == 99 and ignore_graceful_exit_experiments:
            logger.info("found graceful exit (99), not adding to file: %s", named_id)
            continue
        elif result["rc"] == 0:
            test_stats = rename_flatten(result["details"]["test_stats"], "test_stats")
            dev_stats = rename_flatten(result["details"]["dev_stats"], "dev_stats")

            flattened_result = {
                **without_keys(result, ["details"]),
                **dev_stats,
                **test_stats,
            }
        else:
            flattened_result = {**without_keys(result, ["details"])}

        scalared_flattened_result = non_scalar_to_str(flattened_result)
        flattened_results[named_id] = scalared_flattened_result

    df = pd.DataFrame(data=flattened_results.values())

    if num_runs_per_experiment >= 2:
        df_aggr = _aggregate_and_mean(df)
    else:
        df_aggr = None
    del df[COL_NAME_TMP_GROUP]

    _dfs_to_excel(opt.results_path + ".xlsx", {"raw": df, "aggr": df_aggr})


def jsonl2xlsx(opt):
    labels = {2: "positive", 1: "neutral", 0: "negative"}

    with jsonlines.open(opt.results_path, "r") as reader:
        lines = []
        for line in reader:
            if line["true_label"] != line["pred_label"]:
                line["true_label"] = labels[line["true_label"]]
                line["pred_label"] = labels[line["pred_label"]]

                lines.append(line)

        df = pd.DataFrame(data=lines)
        df.to_excel(opt.results_path + ".xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="results/mtscall_stance0",
    )
    parser.add_argument("--mode", type=str, default="shelve")
    opt = parser.parse_args()

    if opt.mode == "shelve":
        shelve2xlsx(opt, ignore_graceful_exit_experiments=False)
    elif opt.mode == "jsonl":
        jsonl2xlsx(opt)
