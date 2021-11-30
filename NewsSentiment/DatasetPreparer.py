"""
Used by controller.py to prepare datasets for an experiment, thereby most importantly considering whether the experiment
uses cross validation or not.

Currently, the class is partially hard coded for the poltsanews dataset.
"""
import math
import os
import random
from collections import Counter
from shutil import copyfile

import jsonlines
from tabulate import tabulate

from NewsSentiment.SentimentClasses import SentimentClasses
from NewsSentiment.fxlogger import get_logger


class DatasetPreparer:
    def __init__(self, name, basepath_datasets, task_format, human, non_human):
        # only used for newsmtsc format
        SentimentClasses.Sentiment3ForNewsMtsc()
        self.basepath_datasets = basepath_datasets
        self.name = name
        self.human_created_filenames = human
        self.non_human_created_filenames = non_human
        self.human_created_filepaths = [
            self.get_filepath_by_name(x) for x in self.human_created_filenames
        ]
        self.non_human_created_filepaths = [
            self.get_filepath_by_name(x) for x in self.non_human_created_filenames
        ]
        self.data_types = ["human", "nonhum"]

        self.sets_info = None

        self.random_seed = 1337
        random.seed(self.random_seed)
        self.logger = get_logger()

        self.examples_human = self.files_to_dictlst(self.human_created_filepaths)
        self.examples_nonhum = self.files_to_dictlst(self.non_human_created_filepaths)

        self.task_format = task_format

        self.logger.info(
            "shuffling example lists with seed {}".format(self.random_seed)
        )
        random.shuffle(self.examples_human)
        random.shuffle(self.examples_nonhum)

        self.logger.info(
            "{} examples read created by humans (from: {})".format(
                len(self.examples_human), self.human_created_filepaths
            )
        )
        self.logger.info(
            "{} examples read not created by humans (from: {})".format(
                len(self.examples_nonhum), self.non_human_created_filepaths
            )
        )

    def get_filepath_by_name(self, filename):
        return os.path.join(self.basepath_datasets, self.name, filename)

    def file_to_dictlst(self, filepath):
        dict_lst = []
        with jsonlines.open(filepath, "r") as reader:
            for line in reader:
                dict_lst.append(line)
        self.logger.debug("{} examples read from {}".format(len(dict_lst), filepath))
        return dict_lst

    def files_to_dictlst(self, list_filepaths):
        dict_lst = []
        for filepath in list_filepaths:
            dict_lst.extend(self.file_to_dictlst(filepath))
        return dict_lst

    def create_slices(self, data_type):
        """

        :param sets_info:
        :param data_type: 'human', 'nonhum'
        :return:
        """
        set_names = list(self.sets_info.keys())
        assert data_type in self.data_types

        # get some vars
        _id_relative_weight = "{}-rel-weight".format(data_type)
        _id_examples = "{}-examples".format(data_type)

        prev_split_pos = 0
        for set_index, set_name in enumerate(set_names):
            cur_setinfo = self.sets_info[set_name]

            if _id_relative_weight in cur_setinfo:
                cur_relative_weight = cur_setinfo[_id_relative_weight]

                if cur_relative_weight:
                    split_pos = prev_split_pos + math.floor(
                        cur_relative_weight * len(self.examples_human)
                    )
                    if set_index == len(set_names) - 1:
                        # just to be sure to not miss a single example because of rounding
                        split_pos = len(self.examples_human)

                    example_slice = self.examples_human[prev_split_pos:split_pos]
                    cur_setinfo[_id_examples] = example_slice
                    self.logger.info(
                        "{}: added {} {} examples ({:.2f})".format(
                            set_name, len(example_slice), data_type, cur_relative_weight
                        )
                    )

                    prev_split_pos = split_pos
                else:
                    cur_setinfo[_id_examples] = []

    def merge_slices(self):
        total_examples_count = 0
        for set_name, cur_set in self.sets_info.items():
            cur_merged_data = []
            for data_type in self.data_types:
                _id_examples = "{}-examples".format(data_type)

                if _id_examples in cur_set:
                    cur_merged_data.extend(cur_set[_id_examples])

            cur_set["examples"] = cur_merged_data
            total_examples_count += len(cur_merged_data)

        for set_name, cur_set in self.sets_info.items():
            cur_set["examples-rel"] = len(cur_set["examples"]) / total_examples_count

    def _get_label_counts(self, tasks):
        label_counts = Counter()
        if self.task_format == "newsmtsc":
            for task in tasks:
                assert len(task["targets"]) >= 1, f"no target in task: {task}"
                for target in task["targets"]:
                    label = SentimentClasses.polarity2label(target["polarity"])
                    label_counts[label] += 1
        elif self.task_format == "newstsc":
            for task in tasks:
                label_counts[task["label"]] += 1
        else:
            raise NotImplementedError(f"unknown data format. {self.task_format}")
        return dict(label_counts)

    def print_set_info(self):
        header = [
            "set_name",
            "human rel",
            "human abs",
            "non-hum rel",
            "non-hum abs",
            "rel",
            "abs",
            "pos",
            "neu",
            "neg",
        ]
        rows = []
        for set_name, cur_set in self.sets_info.items():
            if "file" in cur_set:
                num_lines = sum(
                    1 for line in open(self.get_filepath_by_name(cur_set["file"]), encoding="utf8")
                )
                row = [set_name, -1, -1, -1, -1, cur_set["file"], num_lines, -1, -1, -1]
            else:
                label_counts = self._get_label_counts(
                    [*(cur_set["human-examples"]), *(cur_set["nonhum-examples"])]
                )
                row = [
                    set_name,
                    cur_set["human-rel-weight"],
                    len(cur_set["human-examples"]),
                    cur_set["nonhum-rel-weight"],
                    len(cur_set["nonhum-examples"]),
                    cur_set["examples-rel"],
                    len(cur_set["examples"]),
                    label_counts["positive"],
                    label_counts["neutral"],
                    label_counts["negative"],
                ]
            rows.append(row)

        self.logger.info("\n" + tabulate(rows, header))

    def init_set(self, sets_info):
        self.sets_info = sets_info

        set_names = list(self.sets_info.keys())

        weights_human = Counter()
        weights_nonhum = Counter()
        nonnull_set_names = []
        self.filecopy_sets = {}

        # sum weights and thereby filter datasets that are 0 in size
        for set_name in set_names:
            setinfo = self.sets_info[set_name]

            if "file" in setinfo:
                self.filecopy_sets[set_name] = setinfo["file"]
            else:
                weights_human[set_name] = setinfo["human-weight"]
                weights_nonhum[set_name] = setinfo["nonhum-weight"]

                if setinfo["human-weight"] + setinfo["nonhum-weight"] > 0:
                    nonnull_set_names.append(set_name)
                else:
                    self.logger.info(
                        "discard {}, because would be empty".format(set_name)
                    )
        set_names = nonnull_set_names

        human_weight_sum = sum(weights_human.values())
        nonhuman_weight_sum = sum(weights_nonhum.values())

        # add relative weights
        for set_name in set_names:
            cur_setinfo = self.sets_info[set_name]
            if cur_setinfo["human-weight"]:
                cur_setinfo["human-rel-weight"] = (
                    cur_setinfo["human-weight"] / human_weight_sum
                )
            else:
                cur_setinfo["human-rel-weight"] = 0
            if cur_setinfo["nonhum-weight"]:
                cur_setinfo["nonhum-rel-weight"] = (
                    cur_setinfo["nonhum-weight"] / nonhuman_weight_sum
                )
            else:
                cur_setinfo["nonhum-rel-weight"] = 0
        # at this point, setsinfo contains only positive (non-0) relative weights (or no such key, if the abs. pad_value was 0, too)

        # create slices
        self.create_slices("human")
        self.create_slices("nonhum")

        # merge human and non human sets in each set
        self.merge_slices()

        self.print_set_info()

    def export(self, savepath):
        for set_name, cur_set in self.sets_info.items():
            if "file" not in cur_set:
                set_savefolder = os.path.join(savepath, self.name)
                os.makedirs(set_savefolder, exist_ok=True)
                set_savepath = os.path.join(set_savefolder, set_name + ".jsonl")

                with jsonlines.open(set_savepath, "w") as writer:
                    writer.write_all(cur_set["examples"])
                self.logger.debug(
                    "created set (abs={}) at {}".format(
                        len(cur_set["examples"]), set_savepath
                    )
                )

        for set_name, filename in self.filecopy_sets.items():
            set_savefolder = os.path.join(savepath, self.name)
            os.makedirs(set_savefolder, exist_ok=True)
            set_savepath = os.path.join(set_savefolder, set_name + ".jsonl")
            set_sourcepath = self.get_filepath_by_name(filename)

            copyfile(set_sourcepath, set_savepath)
            self.logger.debug("copied set to {}".format(set_savepath))

    @classmethod
    def poltsanews_rel801010_allhuman(cls, basepath):
        name = "poltsanews"
        task_format = "newstsc_old"
        human_created_filenames = ["human.jsonl"]
        non_human_created_filenames = ["train_20191021_233454.jsonl"]

        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"human-weight": 80, "nonhum-weight": 0},
            "dev": {"human-weight": 10, "nonhum-weight": 0},
            "test": {"human-weight": 10, "nonhum-weight": 0},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def poltsanews_crossval8010_allhuman(cls, basepath):
        name = "poltsanews"
        task_format = "newstsc_old"
        human_created_filenames = ["human.jsonl"]
        non_human_created_filenames = ["train_20191021_233454.jsonl"]

        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "crossval": {"human-weight": 80, "nonhum-weight": 0},
            "test": {"human-weight": 10, "nonhum-weight": 0},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def acl14twitter(cls, basepath):
        name = "acl14twitter"
        task_format = "newsmtsc"
        human_created_filenames = ["train.raw.jsonl"]
        non_human_created_filenames = []

        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"human-weight": 80, "nonhum-weight": 0},
            "dev": {"human-weight": 10, "nonhum-weight": 0},
            "test": {"file": "test.raw.jsonl"},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def semeval14laptops(cls, basepath):
        name = "semeval14laptops"
        task_format = "newsmtsc"
        human_created_filenames = ["Laptops_Train.xml.seg.jsonl"]
        non_human_created_filenames = []

        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"human-weight": 80, "nonhum-weight": 0},
            "dev": {"human-weight": 10, "nonhum-weight": 0},
            "test": {"file": "Laptops_Test_Gold.xml.seg.jsonl"},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def semeval14restaurants(cls, basepath):
        name = "semeval14restaurants"
        task_format = "newsmtsc"
        human_created_filenames = ["Restaurants_Train.xml.seg.jsonl"]
        non_human_created_filenames = []

        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"human-weight": 80, "nonhum-weight": 0},
            "dev": {"human-weight": 10, "nonhum-weight": 0},
            "test": {"file": "Restaurants_Test_Gold.xml.seg.jsonl"},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def newsmtsc_devtest_mt(cls, basepath):
        name = "newsmtsc-train-and-test-mt"
        task_format = "newsmtsc"
        human_created_filenames = ["devtest_mtsc_only.jsonl"]
        non_human_created_filenames = []
        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"file": "train.jsonl"},
            "dev": {"human-weight": 30, "nonhum-weight": 0},
            "test": {"human-weight": 70, "nonhum-weight": 0},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format

    @classmethod
    def newsmtsc_devtest_rw(cls, basepath):
        name = "newsmtsc-train-and-test-rw"
        task_format = "newsmtsc"
        human_created_filenames = ["devtest_mtsc_and_single_primaries.jsonl"]
        non_human_created_filenames = []
        dprep = cls(
            name,
            basepath,
            task_format,
            human_created_filenames,
            non_human_created_filenames,
        )

        sets_info = {
            "train": {"file": "train.jsonl"},
            "dev": {"human-weight": 30, "nonhum-weight": 0},
            "test": {"human-weight": 70, "nonhum-weight": 0},
        }

        dprep.init_set(sets_info)
        return dprep, name, task_format


if __name__ == "__main__":
    DatasetPreparer.sentinews("controller_data/datasets")
