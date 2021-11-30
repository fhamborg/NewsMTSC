from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class RandomMulti(FXBaseModel):
    """

    """

    @staticmethod
    def get_language_models():
        """
        All architecture assumes that at least one model is used so we just require
        bert here for compatibility.
        :return:
        """
        return (BERT_BASE_UNCASED,)

    @staticmethod
    def get_input_field_ids():
        return [
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(RandomMulti, self).__init__()
        self.num_classes = opt.polarities_dim

    def forward(self, inputs):
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        batch_size = text_bert_indices.shape[0]
        num_targets = text_bert_indices.shape[1]

        # get a random tensor
        logits = torch.rand(batch_size, num_targets, self.num_classes)

        return logits
