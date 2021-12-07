import torch

from NewsSentiment.knowledge.knowledgeutils import find_key_original_or_lc
from NewsSentiment.knowledge.nrcemolex.converter import PATH_DICT_NRC_EMOLEX, EMOTION2INDEX
from NewsSentiment.diskdict import DiskDict

__ddict_emolex = DiskDict(PATH_DICT_NRC_EMOLEX)
__ddict_emolex_keys_lower = {k.lower(): v for k, v in __ddict_emolex.items()}
__num_emotions = 10
assert len(__ddict_emolex) == 6468
assert len(EMOTION2INDEX) == __num_emotions


def get_num_nrc_emotions():
    return __num_emotions


def get_nrc_emotions_as_tensor(term: str):
    emotions = find_key_original_or_lc(__ddict_emolex, __ddict_emolex_keys_lower, term)
    tensor_emotions = torch.zeros(__num_emotions, dtype=torch.long)
    for emotion_index in emotions:
        tensor_emotions[emotion_index] = 1
    return tensor_emotions
