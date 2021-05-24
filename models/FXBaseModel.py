import torch.nn as nn

from abc import abstractmethod

from transformers import XLNetModel, AlbertModel, BertModel, RobertaModel
from functools import wraps

from torch.hub import load_state_dict_from_url


class FXBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def get_language_models():
        return

    @staticmethod
    @abstractmethod
    def get_input_field_ids():
        return

    def invoke_language_model(self, lm, input_ids, token_type_ids=None):
        type_lm = type(lm)
        if type_lm == XLNetModel:
            last_hidden_state, mems, all_hidden_states = lm(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
        elif type_lm in [AlbertModel, BertModel, RobertaModel]:
            if token_type_ids is None:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids,
                )
            else:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
        else:
            raise NotImplementedError

        return last_hidden_state


def provide_pretrained(version, pretrained_url):
    """
    Usage:

        @provide_pretrained("v1.0.0", "https://example.com/link/to/state_dict")
        class Example(nn.Module):
            pass
    """
    def decorator(model_class):
        # The actual decorator to use before the class
        wraps(model_class)

        wrapper = __get_pretrained_wrapper_class(model_class)
        wrapper._provide_pretrained_versions[version] = pretrained_url

        return wrapper
    return decorator


def default_pretrained(version):
    """
    Usage:
        @default_pretrained("v1.0.0")
        @provide_pretrained("v1.0.0", "https://example.com/link/to/state_dict")
        class Example(nn.Module):
            pass
    """
    def decorator(model_class):
        # The actual decorator to use before the class
        wraps(model_class)

        wrapper = __get_pretrained_wrapper_class(model_class)
        wrapper._provide_pretrained_default = version

        return wrapper
    return decorator


__pretrained_wrapper_classes = set()


def __get_pretrained_wrapper_class(base_class):
    if base_class in __pretrained_wrapper_classes:
        return base_class

    class PretrainedWrapper(base_class):
        _provide_pretrained_default = None
        _provide_pretrained_versions = {}

        def __init__(self, *args, **kwargs):
            super(base_class, self).__init__(*args, **kwargs)

        def has_pretrained_state_dict(self, version=None):
            version = version or self._provide_pretrained_default
            return version in self._provide_pretrained_versions

        def get_pretrained_state_dict(self, version=None, **kwargs):
            url = self.__provide_pretrained_versions[version or self.__provide_pretrained_default]
            return load_state_dict_from_url(url, **kwargs)

    __pretrained_wrapper_classes.add(PretrainedWrapper)
    return PretrainedWrapper
