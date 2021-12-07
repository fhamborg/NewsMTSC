from argparse import Namespace
from typing import Dict

import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.layers.AggregatorForBert import AggregatorForBert
from NewsSentiment.models.FXBaseModel import FXBaseModel


class NoTargetClsBert(FXBaseModel):
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
        super(NoTargetClsBert, self).__init__()
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.aggregator_for_bert = AggregatorForBert(opt.spc_lm_representation)
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(
            self.language_model.config.hidden_size, opt.polarities_dim
        )

    def forward(self, inputs):
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )

        last_hidden_state, pooler_output, all_hidden_states = self.language_model(
            input_ids=text_bert_indices
        )
        prepared_output = self.aggregator_for_bert(
            last_hidden_state, pooler_output, all_hidden_states
        )

        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits
