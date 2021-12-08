from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class TDBertLikeSingle(FXBaseModel):
    """
    This model returns uses a target mask for a single target to obtain only the hidden
    states of those last layer nodes that correspond to a wordpiece of the target
    phrase. Then aggregation, dropout and a dense layer is applied to retrieve the
    3-class logits.

    From a conceptual perspective, TD-BERT is in some aspects similar (e.g., only the
    output of target-phrase-related nodes is used), but in other aspects not similar,
    e.g., we don't use max pooling but a mean on all non-0 target nodes. In other cases,
    it is not quite clear how the authors of TD-BERT implemented them. An email I sent
    to them was not answered yet.
    """

    @staticmethod
    def get_language_models():
        return (BERT_BASE_UNCASED,)

    @staticmethod
    def get_input_field_ids():
        return [
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(TDBertLikeSingle, self).__init__()
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(
            self.language_model.config.hidden_size, opt.polarities_dim
        )

    def forward(self, inputs):
        # get inputs
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        text_bert_indices_target_mask = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )

        # apply bert
        last_hidden_states, pooler_output, all_hidden_states = self.language_model(
            input_ids=text_bert_indices
        )

        # element-wise multiplication with target mask
        # unsqueeze, cf. https://stackoverflow.com/q/62559382
        text_bert_indices_target_mask_unsqueezed = text_bert_indices_target_mask.unsqueeze(
            -1
        )
        last_hidden_states_only_target = (
            last_hidden_states * text_bert_indices_target_mask_unsqueezed
        )

        # similar to TD-BERT, perform max pooling TODO not implemented yet, instead:
        # for now, retrieve only the values of the target's output tokens and then
        # calculate the mean: (batchsize, 150, 768) -> (batchsize, 768)
        # get the positions of target nodes. note that we cannot simply take the mean
        # as it would divide by the number of the sequence length, whereas the effective
        # length is only of size k, where k is the number of non-zero scalars in the
        # input mask (since we are only interested in those values)
        last_hidden_states_only_target_aggregated = last_hidden_states_only_target.sum(
            dim=1
        )
        # get the sum for each (batch, hidden states) (sum over the sequence length
        # dim)
        denominator_for_mean = text_bert_indices_target_mask_unsqueezed.sum(dim=1)
        # divide for each (batch, hidden states) by the denominator to get the mean
        last_hidden_states_only_target_aggregated = (
            last_hidden_states_only_target_aggregated / denominator_for_mean
        )
        # dropout before dense layer, as in most other tsc models
        last_hidden_states_only_target_aggregated = self.dropout(
            last_hidden_states_only_target_aggregated
        )
        # dense layer
        logits = self.dense(last_hidden_states_only_target_aggregated)

        return logits
