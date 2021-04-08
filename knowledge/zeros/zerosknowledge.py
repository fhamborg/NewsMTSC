import torch


def get_num_zero_dimensions():
    # while one dimension would suffice, using only one can cause (especially when using
    # single_targets=True) that the batcher by pytorch removes the singleton dimension. specifically,
    # with only one target per batch item, the size of the knowledge source will be 1, 150, 1, whereas the last dimension
    # is removed during batching by pytorch. so to keep things simple, we just add a two here to avoid having a
    # singleton dimension at the end
    return 2


def get_zeros_as_tensor(term: str):
    tensor_emotions = torch.zeros(get_num_zero_dimensions(), dtype=torch.long)
    return tensor_emotions
