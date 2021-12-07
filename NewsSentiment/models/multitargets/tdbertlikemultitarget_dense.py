from argparse import Namespace
from typing import Dict

import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class TDBertLikeMultiTargetDense(FXBaseModel):
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
        super(TDBertLikeMultiTargetDense, self).__init__()
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Bilinear(
            self.language_model.config.hidden_size,
            FXDataset.NUM_MAX_TARGETS_PER_ITEM,
            FXDataset.NUM_MAX_TARGETS_PER_ITEM,
        )
        self.dense2 = nn.Linear(opt.max_seq_len, opt.polarities_dim)

    def forward(self, inputs):
        # get inputs
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        text_bert_indices_targets_mask = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        # prepare inputs
        # for text only, we do not need target specific information, i.e., all text
        # vectors are identical. also, bert, can
        # only process one sequence of size max_seq_len (more specifically, a tensor
        # of size batch_size x max_seq_ken). thus, we select only the first element from
        # the second dimension (the dimensions are: batch, targets, hidden_states)
        text_bert_indices = text_bert_indices[:, 0, :]
        # apply bert
        last_hidden_states, pooler_output, all_hidden_states = self.language_model(
            input_ids=text_bert_indices
        )

        # dropout
        last_hidden_states = self.dropout(last_hidden_states)

        # shapes:
        # last_hidden_states:                   batch, seqlen, bertdim
        # text_bert_indices_targets_mask:       batch, target, seqlen
        # new text_bert_indices_targets_mask:   batch, seqlen, target
        text_bert_indices_targets_mask = text_bert_indices_targets_mask.permute(
            0, 2, 1
        ).clone()

        logits = self.dense(last_hidden_states, text_bert_indices_targets_mask)
        logits = logits.permute(0, 2, 1)
        logits = self.dense2(logits)

        return logits
