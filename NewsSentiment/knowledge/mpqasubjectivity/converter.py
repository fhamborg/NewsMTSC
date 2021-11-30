import os
from collections import defaultdict, Counter
from pathlib import Path
import csv

from tqdm import tqdm

from NewsSentiment.fxlogger import get_logger
from NewsSentiment.diskdict import DiskDict

POLARITY2INDEX = {
    "positive": 2,
    "neutral": 1,
    "negative": 0,
}
THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PATH_DICT_MPQA_SUBJECTIVITY = THIS_DIR / "subjclueslen1-HLTEMNLP05.tff.ddict"

logger = get_logger()


def get_value(entry_key_value: str):
    return entry_key_value.split("=")[1]


def convert_txt_to_dict():
    path_dict = THIS_DIR / "subjclueslen1-HLTEMNLP05.tff"

    term2polarity = defaultdict(set)
    polarity_counter = Counter()
    with open(path_dict, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ")
        for line in csv_reader:
            entry_type, entry_length, entry_word, entry_pos, entry_is_stemmed, entry_polarity = line
            entry_type = get_value(entry_type)
            entry_length = get_value(entry_length)
            entry_word = get_value(entry_word)
            entry_pos = get_value(entry_pos)
            entry_is_stemmed = get_value(entry_is_stemmed)
            entry_polarity = get_value(entry_polarity)

            assert " " not in entry_word, f"cannot handle spaces in word"
            assert entry_polarity in POLARITY2INDEX.keys() or entry_polarity == "both", f"polarity label not known: {entry_polarity} for {entry_word}"

            if entry_polarity == "both":
                polarities = [POLARITY2INDEX["positive"], POLARITY2INDEX["negative"]]
            else:
                polarities = (POLARITY2INDEX[entry_polarity],)

            for polarity in polarities:
                term2polarity[entry_word].add(polarity)
                polarity_counter[polarity] += 1

    logger.info("read %s terms", len(term2polarity))
    logger.info("polarity count:\n%s", polarity_counter.most_common())

    logger.info("saving to %s...", PATH_DICT_MPQA_SUBJECTIVITY)
    ddict_emolex = DiskDict(PATH_DICT_MPQA_SUBJECTIVITY)
    ddict_emolex.update(term2polarity)
    ddict_emolex.sync_to_disk()
    logger.info("done")


if __name__ == "__main__":
    convert_txt_to_dict()
