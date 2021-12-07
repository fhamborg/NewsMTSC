from abc import ABC
from argparse import Namespace
from typing import List, Dict
import torch.nn as nn
import torch

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class FXEnsemble(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (
            BERT_BASE_UNCASED,
            ROBERTA_BASE,
            XLNET_BASE_CASED,
        )

    @staticmethod
    def get_input_field_ids():
        return [
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            (ROBERTA_BASE, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (ROBERTA_BASE, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            (XLNET_BASE_CASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (XLNET_BASE_CASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super().__init__()

        # language models
        self.bert = transformer_models[BERT_BASE_UNCASED]
        self.roberta = transformer_models[ROBERTA_BASE]
        self.xlnet = transformer_models[XLNET_BASE_CASED]
        self.num_models = 3

        # params
        assert (
            self.bert.config.hidden_size
            == self.roberta.config.hidden_size
            == self.xlnet.config.hidden_size
        )
        self.sequence_length = opt.max_seq_len
        self.hidden_size = self.bert.config.hidden_size

        # other neural network components
        self.dropout = nn.Dropout(opt.dropout)
        self.target_dependent_text_combiner = nn.Linear(
            self.hidden_size * self.num_models, opt.polarities_dim
        )

    def _combine_text_out_with_target_mask(
        self, batch_size, text_last_hidden_state, target_mask
    ):
        roberta_target_mask = target_mask.reshape((batch_size, 1, self.sequence_length))
        roberta_target_dependent_text = torch.bmm(
            roberta_target_mask, text_last_hidden_state
        )
        roberta_target_dependent_text = roberta_target_dependent_text.reshape(
            (batch_size, self.hidden_size)
        )
        return roberta_target_dependent_text

    def forward(self, inputs: List):
        # alternatively, we could also use this
        # FXDataset.get_all_inputs_for_model(input, self)
        bert_text_ids = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        bert_target_mask = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        roberta_text_ids = FXDataset.get_input_by_params(
            inputs, ROBERTA_BASE, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        roberta_target_mask = FXDataset.get_input_by_params(
            inputs, ROBERTA_BASE, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        xlnet_text_ids = FXDataset.get_input_by_params(
            inputs, XLNET_BASE_CASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        xlnet_target_mask = FXDataset.get_input_by_params(
            inputs, XLNET_BASE_CASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )

        # get variables
        batch_size = bert_text_ids.shape[0]

        # dev notes:
        # batch_size = 4, sequence_length = 150, hidden_size = 768

        # bert_text_out returns list with following elements
        # 0: last_hidden_state (batch_size, sequence_length, hidden_size)
        # 1: pooler_output (batch_size, hidden_size)
        # 2: hidden_states (batch_size, sequence_length, hidden_size)
        (
            bert_text_last_hidden_state,
            bert_text_pooler_output,
            bert_text_hidden_states,
        ) = self.bert(bert_text_ids)
        # roberta_text_out returns same output as bert
        (
            roberta_text_last_hidden_state,
            roberta_text_pooler_output,
            roberta_text_hidden_states,
        ) = self.roberta(roberta_text_ids)
        # xlnet_text_out returns list with following elements
        # 0: last_hidden_state (batch_size, sequence_length, hidden_size)
        # does not exist - (1: mems, a list of length config.n_layers)
        # 2: hidden_states (batch_size, sequence_length, hidden_size)
        xlnet_text_last_hidden_state, xlnet_text_hidden_states = self.xlnet(
            xlnet_text_ids
        )

        # incorporate target masks with (for now) last layer's states
        # *_target_dependent_text_out will be of shape (batch_size, hidden_size)
        bert_target_dependent_text_out = self._combine_text_out_with_target_mask(
            batch_size, bert_text_last_hidden_state, bert_target_mask
        )
        roberta_target_dependent_text_out = self._combine_text_out_with_target_mask(
            batch_size, roberta_text_last_hidden_state, roberta_target_mask
        )
        xlnet_target_dependent_text_out = self._combine_text_out_with_target_mask(
            batch_size, xlnet_text_last_hidden_state, xlnet_target_mask
        )

        # cat outputs
        cat_target_dependent_text_out = torch.cat(
            [
                bert_target_dependent_text_out,
                roberta_target_dependent_text_out,
                xlnet_target_dependent_text_out,
            ],
            dim=1,
        )

        # dropout for better learning
        cat_target_dependent_text_out = self.dropout(cat_target_dependent_text_out)

        # combine and get 3 dimensions
        logits = self.target_dependent_text_combiner(cat_target_dependent_text_out)

        return logits
