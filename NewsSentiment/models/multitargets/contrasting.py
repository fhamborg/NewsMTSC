from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class Contrasting(FXBaseModel):
    """
    This model returns a sequence that only contains the hidden states of those output
    nodes that represent a word piece of the target phrase. All other hidden states are
    set to 0. The length of the output sequence is opt.max_seq_len

    From a conceptual perspective, TD-BERT is in some aspects similar (e.g., only the
    output of target-phrase-related nodes is used), but in other aspects not similar or
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
        super(Contrasting, self).__init__()
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)

        self.contrasting_weight_dense = nn.Linear(opt.max_seq_len, opt.max_seq_len,)
        self.dense = nn.Linear(
            self.language_model.config.hidden_size, opt.polarities_dim
        )

    def forward(self, inputs):
        # get inputs
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        text_bert_indices_targets_mask = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        assert text_bert_indices.shape[1] == 2, "contrasting requires two targets"

        # target a
        a_text_bert_indices = text_bert_indices[:, 0, :]
        a_text_bert_indices_targets_mask = text_bert_indices_targets_mask[:, 0, :]

        # target b
        b_text_bert_indices = text_bert_indices[:, 1, :]
        b_text_bert_indices_targets_mask = text_bert_indices_targets_mask[:, 1, :]

        # bert
        (
            a_last_hidden_states,
            a_pooler_output,
            a_all_hidden_states,
        ) = self.language_model(input_ids=a_text_bert_indices)
        (
            b_last_hidden_states,
            b_pooler_output,
            b_all_hidden_states,
        ) = self.language_model(input_ids=b_text_bert_indices)
        stacked_bert_outs_ab = torch.stack(
            (a_last_hidden_states, b_last_hidden_states), dim=1
        )
        # stacked_bert_outs_ab
        stacked_bert_outs_ab = self.dropout(stacked_bert_outs_ab)
        # shape: batch, 2, seqlen, bertdim

        # create weight
        cross_weight = self.contrasting_weight_dense(text_bert_indices_targets_mask)
        cross_weight = cross_weight.unsqueeze(3).repeat(
            1, 1, 1, stacked_bert_outs_ab.shape[3]
        )
        cross_weight = self.dropout(cross_weight)
        # shape: batch, 2, seqlen

        weighted_stacked_bert_outs_ab = stacked_bert_outs_ab * cross_weight

        # sum
        weighted_stacked_bert_outs_ab = weighted_stacked_bert_outs_ab.sum(dim=2)

        # dense
        logits = self.dense(weighted_stacked_bert_outs_ab)

        return logits, cross_weight
