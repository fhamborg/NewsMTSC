from collections import defaultdict, Counter
from pathlib import Path
import csv
import os

from tqdm import tqdm

from NewsSentiment.fxlogger import get_logger
from NewsSentiment.diskdict import DiskDict

POLARITY2INDEX = {
    "positive": 1,
    "negative": 0,
}
THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PATH_DICT_BING_LIU_OPINION_POLARITY = THIS_DIR / "opinion_polarity.ddict"

logger = get_logger()


def convert_single_txt_to_dict(path: Path):
    terms = set()
    with open(path, "r") as file:
        line_count = 0
        for line in file:
            if line_count > 29:
                terms.add(line.strip())
            line_count += 1
    return terms


def convert_txt_to_dict():
    path_pos = THIS_DIR / "positive-words.txt"
    path_neg = THIS_DIR / "negative-words.txt"

    term2polarity = defaultdict(set)
    polarity_counter = Counter()

    positive_terms = convert_single_txt_to_dict(path_pos)
    negative_terms = convert_single_txt_to_dict(path_neg)
    all_terms = positive_terms.union(negative_terms)

    for term in all_terms:
        if term in positive_terms:
            term2polarity[term].add(POLARITY2INDEX["positive"])
            polarity_counter["positive"] += 1
        if term in negative_terms:
            term2polarity[term].add(POLARITY2INDEX["negative"])
            polarity_counter["negative"] += 1

    logger.info("read %s terms", len(term2polarity))
    logger.info("polarity count:\n%s", polarity_counter.most_common())

    logger.info("saving to %s...", PATH_DICT_BING_LIU_OPINION_POLARITY)
    ddict_emolex = DiskDict(PATH_DICT_BING_LIU_OPINION_POLARITY)
    ddict_emolex.update(term2polarity)
    ddict_emolex.sync_to_disk()
    logger.info("done")


if __name__ == "__main__":
    convert_txt_to_dict()
