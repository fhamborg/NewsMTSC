import torch

from NewsSentiment.knowledge.liwc.liwchelper import load_token_parser

parse, category_names = load_token_parser()

LIWC_CATEGORY2INDEX = {}
for index, category_name in enumerate(category_names):
    LIWC_CATEGORY2INDEX[category_name] = index


def get_num_liwc_categories():
    return len(category_names)


def get_liwc_categories_as_tensor(term: str):
    categories = parse(term)
    categories_of_lowercased = parse(term.lower())

    if len(categories) == 0 and len(categories_of_lowercased) > 0:
        # if we do not have categories of original term, but have them for lowercased term, use the latter
        categories = categories_of_lowercased

    tensor_emotions = torch.zeros(get_num_liwc_categories(), dtype=torch.long)
    for category in categories:
        index = LIWC_CATEGORY2INDEX[category]
        assert index < get_num_liwc_categories()
        tensor_emotions[index] = 1
    return tensor_emotions
