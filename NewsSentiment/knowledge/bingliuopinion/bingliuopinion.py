import torch

from NewsSentiment.diskdict import DiskDict
from NewsSentiment.knowledge.bingliuopinion.converter import PATH_DICT_BING_LIU_OPINION_POLARITY, POLARITY2INDEX
from NewsSentiment.knowledge.knowledgeutils import find_key_original_or_lc

__ddict_emolex = DiskDict(PATH_DICT_BING_LIU_OPINION_POLARITY)
__ddict_emolex_keys_lower = {k.lower(): v for k, v in __ddict_emolex.items()}
__num_emotions = 2
assert len(__ddict_emolex) == 6726
assert len(POLARITY2INDEX) == __num_emotions


def get_num_bingliu_polarities():
    return __num_emotions


def get_bingliu_polarities_as_tensor(term: str):
    emotions = find_key_original_or_lc(__ddict_emolex, __ddict_emolex_keys_lower, term)
    tensor_emotions = torch.zeros(__num_emotions, dtype=torch.long)
    for emotion_index in emotions:
        tensor_emotions[emotion_index] = 1
    return tensor_emotions
