import torch

from NewsSentiment.diskdict import DiskDict
from NewsSentiment.knowledge.knowledgeutils import find_key_original_or_lc
from NewsSentiment.knowledge.mpqasubjectivity.converter import POLARITY2INDEX, PATH_DICT_MPQA_SUBJECTIVITY

__ddict_emolex = DiskDict(PATH_DICT_MPQA_SUBJECTIVITY)
__ddict_emolex_keys_lower = {k.lower(): v for k, v in __ddict_emolex.items()}
__num_emotions = 3
assert len(__ddict_emolex) == 6886, len(__ddict_emolex)
assert len(POLARITY2INDEX) == __num_emotions


def get_num_mpqa_subjectivity_polarities():
    return __num_emotions


def get_mpqa_subjectivity_polarities_as_tensor(term: str):
    emotions = find_key_original_or_lc(__ddict_emolex, __ddict_emolex_keys_lower, term)
    tensor_emotions = torch.zeros(__num_emotions, dtype=torch.long)
    for emotion_index in emotions:
        tensor_emotions[emotion_index] = 1
    return tensor_emotions
