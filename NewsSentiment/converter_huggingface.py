"""
This file converts the dataset files (3 splits) into the format we'll use on
Huggingface Hub, i.e., where examples with k targets are expanded to k examples, each
having 1 target.
"""

import pathlib
import jsonlines
from loguru import logger


def convert_polarity(polarity):
    if polarity == 2.0:
        return -1
    elif polarity == 4.0:
        return 0
    elif polarity == 6.0:
        return 1
    else:
        raise ValueError


def convert_target(obj, target):
    converted_obj = {
        "mention": target["mention"],
        "polarity": convert_polarity(target["polarity"]),
        "from": target["from"],
        "to": target["to"],
        "sentence": obj["sentence_normalized"],
        "id": target["Input.gid"],
    }

    return converted_obj


def convert_obj(obj):
    targets = obj["targets"]
    converted_objs = []

    for target in targets:
        converted_objs.append(convert_target(obj, target))

    return converted_objs


def convert(path):
    files = [p for p in pathlib.Path(path).iterdir() if p.is_file()]

    for file in files:
        converted_lines = []
        counter = 0
        with jsonlines.open(file) as reader:
            for obj in reader:
                converted_lines.extend(convert_obj(obj))
                counter += 1
        logger.info(
            "converted {} lines to {} examples in {}",
            counter,
            len(converted_lines),
            file,
        )

        with jsonlines.open(str(file) + "converted", mode="w") as writer:
            writer.write_all(converted_lines)


if __name__ == "__main__":
    convert("experiments/default/datasets/newsmtsc-mt")
    convert("experiments/default/datasets/newsmtsc-rw")
