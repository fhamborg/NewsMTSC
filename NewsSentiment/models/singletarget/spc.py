from argparse import Namespace
from typing import Dict

import torch.nn as nn

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.layers.AggregatorForBert import AggregatorForBert
from NewsSentiment.models.FXBaseModel import FXBaseModel


class SPC_Base(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (get_default_lm(),)

    @staticmethod
    def get_input_field_ids():
        return [
            (get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (
                get_default_lm(),
                FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
            ),
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(SPC_Base, self).__init__()
        self.language_model = transformer_models[get_default_lm()]
        self.aggregator_for_bert = AggregatorForBert(opt.spc_lm_representation)
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(
            self.language_model.config.hidden_size, opt.polarities_dim
        )

    def forward(self, inputs):
        text_target_bert_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS
        )
        text_target_bert_segments_ids = FXDataset.get_input_by_params(
            inputs,
            get_default_lm(),
            FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
        )

        last_hidden_state = self.invoke_language_model(
            lm=self.language_model,
            input_ids=text_target_bert_indices,
            token_type_ids=text_target_bert_segments_ids,
        )
        # the following two variables can only be derived for some model, whereas invoke_language_model currently
        # returns only one last_hidden_state.
        assert self.aggregator_for_bert.spc_lm_representation == "mean_last"
        pooler_output, all_hidden_states = None, None
        prepared_output = self.aggregator_for_bert(
            last_hidden_state, pooler_output, all_hidden_states
        )
        prepared_output = self.dropout(prepared_output)
        logits = self.dense(prepared_output)

        return logits
