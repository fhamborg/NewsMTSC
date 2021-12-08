from argparse import Namespace
from typing import Dict

import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class TDBertLikeMultiTarget(FXBaseModel):
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
            # (
            #     BERT_BASE_UNCASED,
            #     FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES,
            # ),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(TDBertLikeMultiTarget, self).__init__()
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
        text_bert_indices_targets_mask = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        # text_bert_indices_nrc_emolex = FXDataset.get_input_by_params(
        #     inputs,
        #     BERT_BASE_UNCASED,
        #     FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES,
        # )
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
        # element-wise multiplication with target mask
        # align the dimensions of the tensors
        # last_hidden_states.shape = 4,150,768; should be 4,5,150,768
        # insert a new singleton dimension after the first dimension
        # new shape: 4,1,150,768
        last_hidden_states = last_hidden_states.unsqueeze(1)
        # repeat k times along the new 2nd dimension, where k is the target size
        last_hidden_states = last_hidden_states.repeat(
            1, text_bert_indices_targets_mask.shape[1], 1, 1
        )
        # text_bert_indices_targets_mask.shape = 4,5,150; should be 4,5,150,768
        # insert singleton simension after the three already existing dimensions
        text_bert_indices_targets_mask = text_bert_indices_targets_mask.unsqueeze(3)
        text_bert_indices_targets_mask = text_bert_indices_targets_mask.repeat(
            1, 1, 1, last_hidden_states.shape[3]
        )

        last_hidden_states_only_targets = (
            last_hidden_states * text_bert_indices_targets_mask
        )

        # similar to TD-BERT, perform max pooling TODO not implemented yet, instead:
        # for now, retrieve only the values of the target's output tokens and then
        # calculate the mean:
        # (batchsize, targetsize, 150, 768) -> (batchsize, targetsize, 768)
        # get the positions of target nodes. note that we cannot simply take the mean
        # as it would divide by the number of the sequence length, whereas the effective
        # length is only of size k, where k is the number of non-zero scalars in the
        # input mask (since we are only interested in those values)
        last_hidden_states_aggregated_per_target = last_hidden_states_only_targets.sum(
            dim=2
        )
        # get the sum for each (batch, hidden states) (sum over the sequence length
        # dim)
        denominator_for_mean = text_bert_indices_targets_mask.sum(dim=2)
        # divide for each (batch, hidden states) by the denominator to get the mean
        last_hidden_states_aggregated_per_target = (
            last_hidden_states_aggregated_per_target / denominator_for_mean
        )
        # dropout before dense layer, as in most other tsc models
        last_hidden_states_aggregated_per_target = self.dropout(
            last_hidden_states_aggregated_per_target
        )
        # dense layer
        logits = self.dense(last_hidden_states_aggregated_per_target)

        return logits
