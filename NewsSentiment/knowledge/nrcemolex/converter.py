import os
from collections import defaultdict, Counter
from pathlib import Path
import csv

from NewsSentiment.fxlogger import get_logger
from NewsSentiment.diskdict import DiskDict

EMOTION2INDEX = {
    "anger": 0,
    "anticipation": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "negative": 5,
    "positive": 6,
    "sadness": 7,
    "surprise": 8,
    "trust": 9,
}
THIS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PATH_DICT_NRC_EMOLEX = THIS_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt.ddict"

logger = get_logger()


def convert_txt_to_dict():
    path_dict = THIS_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    term2emotions = defaultdict(set)
    emotion_counter = Counter()
    with open(path_dict, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for term, emotion, is_present in csv_reader:
            if is_present == "1":
                emotion_index = EMOTION2INDEX[emotion]
                term2emotions[term].add(emotion_index)
                emotion_counter[emotion] += 1
            elif is_present == "0":
                pass
            else:
                raise ValueError
    logger.info("read %s terms", len(term2emotions))
    logger.info("emotion count:\n%s", emotion_counter.most_common())

    logger.info("saving to %s...", None)
    ddict_emolex = DiskDict(PATH_DICT_NRC_EMOLEX)
    ddict_emolex.update(term2emotions)
    ddict_emolex.sync_to_disk()
    logger.info("done")


if __name__ == "__main__":
    convert_txt_to_dict()
