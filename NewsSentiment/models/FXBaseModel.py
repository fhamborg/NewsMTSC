import os
from copy import copy

import torch.nn as nn

from abc import abstractmethod

from transformers import (
    XLNetModel,
    AlbertModel,
    BertModel,
    RobertaModel,
    PreTrainedModel,
)
from functools import wraps

from NewsSentiment.download import Download


class FXBaseModel(PreTrainedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                input_ids=input_ids, token_type_ids=token_type_ids,
            )
        elif type_lm in [AlbertModel, BertModel, RobertaModel]:
            if token_type_ids is None:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids,
                )
            else:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids, token_type_ids=token_type_ids
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
    Set the version which should be used as the default version and will be used when running with --pretrained.

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


def model_includes_pretrained(model):
    """
    Checks if a model-class includes the methods to load pretrained models.

    Arguments:
        model Model-class to check.

    Returns:
        True if it includes the functionality.
    """
    return hasattr(model, "has_pretrained_state_dict") and hasattr(
        model, "get_pretrained_state_dict"
    )


__pretrained_wrapper_classes = set()


def __get_pretrained_wrapper_class(base_class):
    if base_class in __pretrained_wrapper_classes:
        return base_class

    class PretrainedWrapper(base_class):
        _provide_pretrained_default = None
        _provide_pretrained_versions = {}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def has_pretrained_state_dict(cls, version=None):
            version = version or cls._provide_pretrained_default
            return version in cls._provide_pretrained_versions

        @classmethod
        def get_pretrained_state_dict(
            cls, version=None, download_if_not_exists=True, **kwargs
        ):
            path = Download.model_path(cls, version)
            if os.path.isfile(path):
                if download_if_not_exists:
                    Download.download(cls, version, False)
                else:
                    raise FileNotFoundError("State dict not found")
            return cls.load_state_dict(path, **kwargs)

        @classmethod
        def get_pretrained_versions(cls):
            return copy(cls._provide_pretrained_versions)

        @classmethod
        def get_pretrained_source(cls, version=None):
            return cls._provide_pretrained_versions[
                version or cls._provide_pretrained_default
            ]

        @classmethod
        def get_pretrained_default_version(cls):
            return cls._provide_pretrained_default

    __pretrained_wrapper_classes.add(PretrainedWrapper)
    return PretrainedWrapper
