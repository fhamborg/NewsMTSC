"""
Definitions:
setup: represents a fixed, unique combination of for instance:
* model, e.g., BERT, DISTILBERT
* hyper-parameters, such as learning rate loss function, weights of classes for loss function
* others, such as softmax,
* input style, e.g., input style: QA style, AEN, ...

Procedure:
Given a single annotated dataset named alldat (also called both.jsonl, since it consists of both splits of the devtest
set):
* splits alldat into testdat and remainderdat

Given a set of experiment setup descriptions
1) Iterate setup descriptions, for each:
a) create 10-fold CV to well evaluate the performance of that particular setup
b) return model from best epoch (or from best fold???!?)
2) retrievel all best performing models and test them on testdat
3) return model that performs best on testdat
"""
from threading import Lock
import argparse
import multiprocessing
import os
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import product

from jsonlines import jsonlines
from tabulate import tabulate
from tqdm import tqdm

from NewsSentiment.combinations_ensemble import combinations_ensemble
from NewsSentiment.combinations_final_lcf2 import combinations_final_lcf2
from NewsSentiment.combinations_final_nostance import combinations_final_nostance
from NewsSentiment.combinations_final_spc import combinations_final_spc
from NewsSentiment.combinations_final_stance0 import combinations_final_stance0
from NewsSentiment.combinations_final_stance1 import combinations_final_stance1
from NewsSentiment.combinations_final_tdbert import combinations_final_tdbert
from NewsSentiment.combinations_top import combinations_top
from NewsSentiment.DatasetPreparer import DatasetPreparer
from NewsSentiment.combinations_default import combinations_default_0
from NewsSentiment.combinations_top4otherdomains import combinations_top4otherdomains
from NewsSentiment.combinations_top_coref16 import combinations_top_coref16
from NewsSentiment.combinations_top_coref8 import combinations_top_coref8
from NewsSentiment.combinations_top_coref8_stanceno import combinations_top_coref8_stanceno
from NewsSentiment.combinations_top_finetuned import combinations_top_finetuned
from NewsSentiment.diskdict import DiskDict
from NewsSentiment.fxlogger import get_logger
from NewsSentiment.train import str2bool

completed_tasks = None  # will be DiskDict later
completed_tasks_in_this_run_count = 0
lock = Lock()


def start_worker(
    experiment_id,
    experiment_named_id,
    named_combination,
    cmd,
    human_cmd,
    experiment_path,
    running_processes,
):
    if "--device" in cmd:
        device_index_in_list = cmd.index("--device")
        device = cmd[device_index_in_list + 1]
    else:
        device = -1

    running_processes[experiment_id] = (True, device)
    logger = get_logger()

    logger.debug("starting single setup: {}".format(human_cmd))
    with open(os.path.join(experiment_path, "stdlog.out"), "w") as file_stdout, open(
        os.path.join(experiment_path, "stdlog.err"), "w"
    ) as file_stderr:
        completed_process = subprocess.run(cmd, stdout=file_stdout, stderr=file_stderr)

    experiment_details = get_experiment_result_detailed(experiment_path)

    running_processes[experiment_id] = (False, device)
    return {
        **named_combination,
        **{"rc": completed_process.returncode, "experiment_id": experiment_id},
        "details": experiment_details,
        "experiment_named_id": experiment_named_id,
    }


def on_task_done(x):
    with lock:
        # result_list is modified only by the main process, not the pool workers.
        completed_tasks[x["experiment_named_id"]] = x
        completed_tasks.sync_to_disk()
        global completed_tasks_in_this_run_count
        completed_tasks_in_this_run_count += 1


def on_task_error(x):
    with lock:
        # result_list is modified only by the main process, not the pool workers.
        completed_tasks[x["experiment_named_id"]] = x
        completed_tasks.sync_to_disk()
        global completed_tasks_in_this_run_count
        completed_tasks_in_this_run_count += 1


def get_experiment_result_detailed(experiment_path):
    experiment_results_path = os.path.join(experiment_path, "experiment_results.jsonl")
    try:
        with jsonlines.open(experiment_results_path, "r") as reader:
            lines = []
            for line in reader:
                lines.append(line)
            assert len(lines) == 1
        return lines[0]
    except FileNotFoundError:
        return None


class SetupController:
    def __init__(self, options):
        self.logger = get_logger()
        self.opt = options

        if self.opt.cuda_devices:
            # to run on SCC
            if self.opt.cuda_devices == "SGE_GPU":
                self.cuda_devices = os.environ.get("SGE_GPU")
            else:
                self.cuda_devices = self.opt.cuda_devices

            if self.cuda_devices:
                self.logger.info("cuda devices:" + self.cuda_devices)
                self.cuda_devices = self.cuda_devices.split(",")
                self.logger.info(
                    f"was assigned {len(self.cuda_devices)} cuda devices: "
                    f"{self.cuda_devices}"
                )
                if self.opt.num_workers < 0:
                    self.logger.info(
                        f"num_workers < 0: using cuda device count. setting "
                        f"num_workers={len(self.cuda_devices)}"
                    )
                    self.opt.num_workers = len(self.cuda_devices)

        else:
            # do not use CUDA
            self.cuda_devices = None

        self.snem = "f1_macro"
        self.experiment_base_path = self.opt.experiments_path

        args_names_ordered = [
            "own_model_name",
            "optimizer",
            "initializer",
            "learning_rate",
            "batch_size",
            "balancing",
            "num_epoch",
            "loss",
            "eval_only_after_last_epoch",
            "devmode",
            "local_context_focus",
            "SRD",
            "pretrained_model_name",
            "state_dict",
            "single_targets",
            "multi_targets",
            "targetclasses",
            "knowledgesources",
            "is_use_natural_target_phrase_for_spc",
            "default_lm",
            "coref_mode_in_training",
        ]

        combinations = None
        if self.opt.combi_mode == "default":
            combinations = combinations_default_0
        elif self.opt.combi_mode == "top":
            combinations = combinations_top
        elif self.opt.combi_mode == "top_coref8":
            combinations = combinations_top_coref8
        elif self.opt.combi_mode == "top_coref8_stanceno":
            combinations = combinations_top_coref8_stanceno
        elif self.opt.combi_mode == "top_coref16":
            combinations = combinations_top_coref16
        elif self.opt.combi_mode == "top_finetuned":
            combinations = combinations_top_finetuned
        elif self.opt.combi_mode == "top4otherdomains":
            combinations = combinations_top4otherdomains
        elif self.opt.combi_mode == "ensemble":
            combinations = combinations_ensemble
        elif self.opt.combi_mode == "final_stance0":
            combinations = combinations_final_stance0
        elif self.opt.combi_mode == "final_stance1":
            combinations = combinations_final_stance1
        elif self.opt.combi_mode == "final_nostance":
            combinations = combinations_final_nostance
        elif self.opt.combi_mode == "final_tdbert":
            combinations = combinations_final_tdbert
        elif self.opt.combi_mode == "final_lcf2":
            combinations = combinations_final_lcf2
        elif self.opt.combi_mode == "final_spc":
            combinations = combinations_final_spc

        if not combinations:
            raise ValueError(
                "combination(mode={}, id={}) not defined".format(
                    self.opt.combi_mode, self.opt.combi_id
                )
            )

        # key: name of parameter that is only applied if its conditions are met
        # pad_value: list of tuples, consisting of parameter name and the pad_value it needs to have in order for the
        # condition to be satisfied
        # Note that all tuples in this list are OR connected, so if at least one is satisfied, the conditions are met.
        # If we need AND connected conditions, my idea is to add an outer list, resulting in a list of lists (of
        # tuples) where all lists are AND connected.
        # If a condition is not satisfied, the corresponding parameter will still be pass
        # if a key is not listed here, there are no conditions and thus the key / parameter will be used always
        conditions = {
            "is_use_natural_target_phrase_for_spc": [("own_model_name", "spc_bert")],
            "spc_lm_representation": [("own_model_name", "spc_bert"),],
            "spc_input_order": [("own_model_name", "spc_bert"),],
            "aen_lm_representation": [("own_model_name", "aen_bert"),],
            "use_early_stopping": [("num_epoch", "10"),],
            "local_context_focus": [
                ("own_model_name", "lcf_bert"),
                ("own_model_name", "lcf_bert2"),
                ("own_model_name", "lcfs_bert"),
                ("own_model_name", "lcfst_bert"),
            ],
            "SRD": [
                ("own_model_name", "lcf_bert"),
                ("own_model_name", "lcf_bert2"),
                ("own_model_name", "lcfs_bert"),
                ("own_model_name", "lcfst_bert"),
            ],
            "single_targets": [
                # we need single_targets mode only for models that are single target
                ("own_model_name", "lcf_bert"),
                ("own_model_name", "lcf_bert2"),
                ("own_model_name", "lcfs_bert"),
                ("own_model_name", "lcft_bert"),
                ("own_model_name", "lcfst_bert"),
                ("own_model_name", "aen_bert"),
                ("own_model_name", "spc_bert"),
                ("own_model_name", "tdbert"),
                ("own_model_name", "EnsembleTopA"),
                ("own_model_name", "EnsembleTopB"),
                ("own_model_name", "tdbert-qa-mul"),
                ("own_model_name", "tdbert-qa-con"),
                ("own_model_name", "tdbertlikesingle"),
                ("own_model_name", "notargetclsbert"),
                ("own_model_name", "random_single"),
                ("own_model_name", "grutsc"),
            ],
            "multi_targets": [
                ("own_model_name", "tdbertlikemulti"),
                ("own_model_name", "tdbertlikemulti_dense"),
                ("own_model_name", "seq2seq"),
                ("own_model_name", "seq2seq_withouttargetmask"),
                ("own_model_name", "random_multi"),
                ("own_model_name", "contrasting"),
            ],
            "knowledgesources": [
                ("own_model_name", "EnsembleTopA"),
                ("own_model_name", "EnsembleTopB"),
                ("own_model_name", "grutsc"),
            ],
            "pretrained_model_name": [("default_lm", "bert-base-uncased"),],
        }

        assert set(args_names_ordered) == set(combinations.keys()), (
            f"mismatch args_names_ordered vs. combinations: "
            f"{set(args_names_ordered).symmetric_difference(combinations)}"
        )

        self.experiment_base_id = (
            self.opt.dataset + "_" + datetime.today().strftime("%Y%m%d-%H%M%S")
        )
        self.basecmd = ["python", "train.py"]
        self.basepath = "controller_data"
        self.basepath_data = os.path.join(self.basepath, "datasets")

        combination_count = 1
        _combination_values = []
        for arg_name in args_names_ordered:
            arg_values = list(combinations[arg_name])
            combination_count = combination_count * len(arg_values)
            _combination_values.append(arg_values)

        combinations = list(product(*_combination_values))
        assert len(combinations) == combination_count

        self.logger.info(
            "{} arguments, totaling in {} combinations".format(
                len(args_names_ordered), combination_count
            )
        )

        # apply conditions
        self.logger.info("applying conditions...")
        self.named_combinations, count_duplicates = self._apply_conditions(
            combinations, args_names_ordered, conditions
        )
        self.logger.info(
            "applied conditions. removed {} combinations. {} -> {}".format(
                count_duplicates, combination_count, len(self.named_combinations)
            )
        )
        self.combination_count = len(self.named_combinations)

        # expand by number of runs per experiment
        expanded_named_combinations = []
        for named_combination in self.named_combinations:
            for run_id in range(self.opt.num_runs_per_experiment):
                expanded_named_combination = named_combination.copy()
                expanded_named_combination["run_id"] = run_id
                expanded_named_combinations.append(expanded_named_combination)
        self.logger.info(
            "expanded to %s runs per experiment. combination count: %s -> %s",
            self.opt.num_runs_per_experiment,
            self.combination_count,
            len(expanded_named_combinations),
        )
        self.named_combinations = expanded_named_combinations
        self.combination_count = len(self.named_combinations)

        if self.opt.dataset == "semeval14restaurants":
            (
                self.dataset_preparer,
                self.datasetname,
                self.task_format,
            ) = DatasetPreparer.semeval14restaurants(self.basepath_data)
        elif self.opt.dataset == "semeval14laptops":
            (
                self.dataset_preparer,
                self.datasetname,
                self.task_format,
            ) = DatasetPreparer.semeval14laptops(self.basepath_data)
        elif self.opt.dataset == "acl14twitter":
            (
                self.dataset_preparer,
                self.datasetname,
                self.task_format,
            ) = DatasetPreparer.acl14twitter(self.basepath_data)
        elif (
            self.opt.dataset
            == "newsmtsc_devtest_rw"
        ):
            (
                self.dataset_preparer,
                self.datasetname,
                self.task_format,
            ) = DatasetPreparer.newsmtsc_devtest_rw(
                self.basepath_data
            )
        elif (
                self.opt.dataset
                == "newsmtsc_devtest_mt"
        ):
            (
                self.dataset_preparer,
                self.datasetname,
                self.task_format,
            ) = DatasetPreparer.newsmtsc_devtest_mt(
                self.basepath_data
            )
        else:
            raise Exception("unknown dataset: {}".format(self.opt.dataset))

    def _apply_conditions(self, combinations, args_names_ordered, conditions):
        named_combinations = []
        seen_experiment_ids = set()
        count_duplicates = 0

        with tqdm(total=len(combinations)) as pbar:
            for combination in combinations:
                named_combination = {}
                full_named_combination = self._args_combination_to_single_arg_values(
                    combination, args_names_ordered
                )

                # for a parameter combination, pass only those parameters that are valid
                # for that combination
                for arg_index, arg_name in enumerate(args_names_ordered):
                    # iterate each parameter and validate - using the other parameter
                    # names and values - whether its conditions are met
                    if self._check_conditions(
                        arg_name, full_named_combination, conditions
                    ):
                        # if yes, pass it
                        named_combination[arg_name] = combination[arg_index]
                        self.logger.debug(
                            "using '{}' in combination {}".format(arg_name, combination)
                        )
                    else:
                        self.logger.debug(
                            "not using '{}' in combination {}".format(
                                arg_name, combination
                            )
                        )

                # check if experiment_id of named combination was already seen
                experiment_id = self._experiment_named_id_from_named_combination(
                    named_combination
                )
                if experiment_id not in seen_experiment_ids:
                    seen_experiment_ids.add(experiment_id)
                    named_combinations.append(named_combination)
                else:
                    count_duplicates += 1

                pbar.update(1)

        return named_combinations, count_duplicates

    def _check_conditions(self, arg_name, full_named_combination, conditions):
        """
        For a given parameter, checks whether its conditions are satisfied. If so, returns True, else False.
        :param arg_name:
        :param arg_value:
        :return:
        """
        if arg_name in conditions and len(conditions[arg_name]) >= 1:
            # at this point we know that there are conditions for the given parameters
            or_connected_conditions = conditions[arg_name]

            for cond_tup in or_connected_conditions:
                cond_param_name = cond_tup[0]
                cond_param_value = cond_tup[1]

                # get parameter and its pad_value in current combination
                if full_named_combination[cond_param_name] == cond_param_value:
                    return True

            # since there was at least one condition due to our check above, we return
            # False here, since the for loop did not return True
            return False

        else:
            # if there is no condition associated with arg_name just return true
            return True

    def _build_args(self, named_args):
        args_list = []
        for arg_name, arg_val in named_args.items():
            self._add_arg(args_list, arg_name, arg_val)

        return args_list

    def _add_arg(self, args_list, name, value):
        args_list.append("--" + name)
        args_list.append(str(value))
        return args_list

    def _prepare_experiment_env(self, experiment_path):
        os.makedirs(experiment_path, exist_ok=True)
        self.dataset_preparer.export(os.path.join(experiment_path, "datasets"))

    def _args_combination_to_single_arg_values(
        self, args_combination, args_names_ordered
    ):
        args_names_values = {}
        for arg_index, arg_name in enumerate(args_names_ordered):
            args_names_values[arg_name] = args_combination[arg_index]
        return args_names_values

    def _experiment_named_id_from_named_combination(self, named_combination):
        return "__".join(["{}={}".format(k, v) for (k, v) in named_combination.items()])

    def prepare_single_setup(self, named_combination, experiment_id, pool_index):
        experiment_path = "{}/{}/".format(self.experiment_base_id, experiment_id)
        experiment_path = os.path.join(self.experiment_base_path, experiment_path)

        self._prepare_experiment_env(experiment_path)

        args = self._build_args(named_combination)
        self._add_arg(args, "snem", self.snem)
        self._add_arg(args, "dataset_name", self.datasetname)
        self._add_arg(args, "experiment_path", experiment_path)
        self._add_arg(args, "data_format", self.task_format)

        if self.cuda_devices:
            # we use the pool_index to get the respective pool's cuda device
            cuda_device_id = self.cuda_devices[pool_index]
            cuda_device_name = "cuda:" + str(cuda_device_id)
            self._add_arg(args, "device", cuda_device_name)

        cmd = self.basecmd + args
        human_cmd = " ".join(cmd)

        with open(os.path.join(experiment_path, "experiment_cmd.sh"), "w") as writer:
            writer.write(human_cmd)

        return cmd, human_cmd, experiment_path

    def run(self):
        global completed_tasks

        if not self.opt.results_path:
            results_path = "results/default_results_{}".format(self.datasetname)
        else:
            results_path = self.opt.results_path

        if not self.opt.continue_run:
            self.logger.info("not continuing")
            os.remove(results_path)
        else:
            self.logger.info("continuing previous run(s)")

        completed_tasks = DiskDict(results_path)
        self.logger.info("found {} previous results".format(len(completed_tasks)))

        self.logger.info("preparing experiment setups...")
        experiment_descs = defaultdict(list)
        previous_tasks = Counter()
        with tqdm(total=self.combination_count) as pbar:
            for i, named_combination in enumerate(self.named_combinations):
                _experiment_named_id = self._experiment_named_id_from_named_combination(
                    named_combination
                )
                # i modulo the number of workers determines which pool this experiment
                # will run in. the pool_index will also later be used to determine
                # which cuda device (if any) to use, thereby making sure, that all tasks
                # of one pool use the same device, whereas each device will be used
                # solely by only one pool (this way, we make sure that at no time two
                # or more tasks will run that use the same cuda device)
                pool_index = i % self.opt.num_workers

                if _experiment_named_id in completed_tasks:
                    task_desc = completed_tasks[_experiment_named_id]

                    if task_desc["rc"] != 0:
                        if self.opt.rerun_non_rc0:
                            self.logger.debug(
                                "task {} was already executed, "
                                "but with rc={}. rerunning.".format(
                                    _experiment_named_id, task_desc["rc"]
                                )
                            )
                            (
                                cmd,
                                human_cmd,
                                experiment_path,
                            ) = self.prepare_single_setup(
                                named_combination, i, pool_index
                            )

                            experiment_descs[pool_index].append(
                                (
                                    i,
                                    _experiment_named_id,
                                    named_combination,
                                    cmd,
                                    human_cmd,
                                    experiment_path,
                                )
                            )

                            previous_tasks["new"] += 1
                            del completed_tasks[_experiment_named_id]
                        else:
                            self.logger.debug(
                                "task {} was already executed, but with rc={}. not "
                                "rerunning.".format(
                                    _experiment_named_id, task_desc["rc"]
                                )
                            )
                            previous_tasks["rcnon0"] += 1
                    else:
                        # rerun tasks where the rc != 0 (always rerun tasks that have
                        # not been executed at all, yet)
                        self.logger.debug(
                            "skipping experiment: {}".format(_experiment_named_id)
                        )
                        self.logger.debug(
                            "previous result: {}".format(
                                completed_tasks[_experiment_named_id]
                            )
                        )
                        previous_tasks["rc0"] += 1
                else:
                    (cmd, human_cmd, experiment_path,) = self.prepare_single_setup(
                        named_combination, i, pool_index
                    )

                    experiment_descs[pool_index].append(
                        (
                            i,
                            _experiment_named_id,
                            named_combination,
                            cmd,
                            human_cmd,
                            experiment_path,
                        )
                    )
                    previous_tasks["new"] += 1
                pbar.update(1)

        self.logger.info(
            "summary (new is also increased for tasks that were executed previously "
            "but yielded rc!=0 - if rerun_non_rc0==True"
        )
        self.logger.info("{}".format(previous_tasks))

        self.logger.info("starting {} experiments".format(self.combination_count))
        self.logger.info(
            "creating {} process pools with each 1 worker".format(self.opt.num_workers)
        )

        if self.cuda_devices and len(self.cuda_devices) != self.opt.num_workers:
            self.logger.warning(
                "number of cuda devices does not match number of workers: "
                "{} vs. {}".format(len(self.cuda_devices), self.opt.num_workers)
            )

        manager = multiprocessing.Manager()
        running_processes = manager.dict()
        for pool_index in range(self.opt.num_workers):
            pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)
            for desc in experiment_descs[pool_index]:
                proc_args = desc + (running_processes,)  # must be a tuple
                pool.apply_async(
                    start_worker,
                    args=proc_args,
                    callback=on_task_done,
                    error_callback=on_task_error,
                )
            self.logger.info(
                f"created pool with {len(experiment_descs[pool_index])} tasks, each "
                f"processed by 1 worker"
            )

        self.logger.info("waiting for workers to complete all jobs...")
        prev_count_done = 0
        with tqdm(total=previous_tasks["new"], initial=prev_count_done) as pbar:
            while completed_tasks_in_this_run_count < previous_tasks["new"]:
                time.sleep(10)
                update_inc = completed_tasks_in_this_run_count - prev_count_done

                if update_inc > 0:
                    pbar.update(update_inc)
                    prev_count_done = completed_tasks_in_this_run_count
                    completed_tasks.sync_to_disk()

                best_dev_snem, best_experiment_id = self._get_best_dev_snem()
                sorted_running_procs = sorted(
                    self._get_running_processes(running_processes),
                    key=lambda running_device: running_device[1],
                )
                pbar.set_postfix_str(
                    f"dev-snem: {best_dev_snem:.3f} ({best_experiment_id}); prcs: {sorted_running_procs}"
                )

        # save to disk
        completed_tasks.sync_to_disk()
        self.logger.info(
            f"finished all tasks ({completed_tasks_in_this_run_count} of "
            f"{previous_tasks['new']})"
        )

        # clone the results so that we can process them while not modifying the original
        processed_results = dict(completed_tasks)

        experiments_rc_overview = Counter()
        non_okay_experiment_ids = []
        for experiment_named_id, experiment_result in processed_results.items():
            rc = experiment_result["rc"]
            experiments_rc_overview[rc] += 1

            if rc != 0:
                if rc == 99:
                    pass
                else:
                    non_okay_experiment_ids.append(experiment_result["experiment_id"])

        self.logger.warning(
            f"{len(non_okay_experiment_ids)} experiments did not return 0: "
            f"{sorted(non_okay_experiment_ids)}"
        )

        # snem-based performance sort
        sorted_results = list(dict(processed_results).values())
        for result in sorted_results:
            if result["details"]:
                result["dev_snem"] = result["details"]["dev_stats"][self.snem]
                del result["details"]
            else:
                result["dev_snem"] = -1.0
        sorted_results.sort(key=lambda x: x["dev_snem"], reverse=True)
        headers = list(sorted_results[0].keys())
        rows = [x.values() for x in sorted_results]

        self.logger.info("all experiments finished. statistics:")
        self.logger.debug("snem-based performances:")
        self.logger.debug("\n" + tabulate(rows, headers))
        self.logger.info("return codes: {}".format(experiments_rc_overview))
        self.logger.info(
            "best dev-snem: %s (id: %s)",
            sorted_results[0]["dev_snem"],
            sorted_results[0]["experiment_id"],
        )

    def _get_best_dev_snem(self):
        best_dev_snem = -1.0
        experiment_id = -1
        # to avoid that the dict is modified while iterating it, copy it for temporary use
        # if we're unlucky this could happen otherwise: RuntimeError: dictionary changed size during iteration
        tmp_tasks = completed_tasks.copy()
        for task in tmp_tasks.values():
            if task.get("details") and task["details"].get("dev_stats"):
                best_dev_snem = max(
                    best_dev_snem, task["details"]["dev_stats"][self.snem]
                )
                experiment_id = task["experiment_id"]
        return best_dev_snem, experiment_id

    def _get_running_processes(self, running_processes):
        return [
            (exp_id, running_device[1])
            for exp_id, running_device in running_processes.items()
            if running_device[0]
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--experiments_path", default="./experiments", type=str)
    parser.add_argument(
        "--continue_run", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "--rerun_non_rc0", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--combi_mode", type=str, required=True)
    parser.add_argument("--combi_id", type=int, default=0)
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="Comma separated list of cuda device IDs, e.g., "
        "0,1,2,3; or SGE_GPU to read this list from "
        "the environment variable SGE_GPU. If not defined,"
        "CPU will be used instead of GPU.",
    )
    parser.add_argument("--num_runs_per_experiment", type=int, default=1)
    parser.add_argument(
        "--only_calc_combinations", type=str2bool, nargs="?", const=True, default=False
    )
    opt = parser.parse_args()

    if opt.only_calc_combinations:
        SetupController(opt)
    else:
        SetupController(opt).run()
