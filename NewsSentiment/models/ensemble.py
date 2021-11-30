from argparse import Namespace
from typing import List, Dict
from NewsSentiment.consts import *
import torch.nn as nn
import torch

from NewsSentiment.models.FXBaseModel import FXBaseModel
from NewsSentiment.models.singletarget.td_bert import TD_BERT


class EnsembleTopA(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (get_default_lm(),)

    @staticmethod
    def get_input_field_ids():
        return [
            # tdbert
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            # hosseinia
            (get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS),
            # while we use text-then-target as bert input, we can use text targetmask and text knowledge source mask
            # because it is identical to a hypothetical text-then-target target mask or text-then-target knowledge
            # source mask (we would not highlight the target in the 2nd component in the corresponding mask)
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super().__init__()
        from models.singletarget.stancedetectionpaper import StanceDetectionHosseinia

        # submodels models
        self.td_bert = TD_BERT(transformer_models, opt)
        self.hosseinia = StanceDetectionHosseinia(transformer_models, opt)

        # ensemble related
        self.hosseinia_dense = nn.Linear(
            self.hosseinia.language_model.config.hidden_size * 3 * 2 * 2,
            self.hosseinia.language_model.config.hidden_size
        )
        self.ensemble_combiner = nn.Linear(
            self.hosseinia.language_model.config.hidden_size * 2,
            opt.polarities_dim
        )

    def forward(self, inputs: List):
        # shape: batch, bertdim
        td_bert_out = self.td_bert(inputs, is_return_ensemble_values=True)
        # shape: batch, 3 * 2 * 2 * bertdim
        hosseinia_out = self.hosseinia(inputs, is_return_ensemble_values=True)

        # to ensure that both models have more or less similar impact on the result, apply a dense layer to hosseinia
        # so that its new shape is: batch, bertdim
        hosseinia_out = self.hosseinia_dense(hosseinia_out)

        # combine
        combined_out = torch.cat((td_bert_out, hosseinia_out), dim=1)

        logits = self.ensemble_combiner(combined_out)

        return logits
