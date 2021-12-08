# this file re-implements TD-BERT by Gao Zhengjie et al.
# while this file aims to be conceptually identical to TD-BERT, one technical difference is that we do not calculate
# the target mask within the model (here) but do this step as part of the dataset processing. in case there are strong
# performance differences between original TD-BERT and this implementation, this technical difference might be worth
# exploring whether it actually yields an identical implementation.
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class TD_BERT(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (get_default_lm(),)

    @staticmethod
    def get_input_field_ids():
        return [
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(TD_BERT, self).__init__()
        self.opt = opt
        self.language_model = transformer_models[get_default_lm()]
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(self.language_model.config.hidden_size, opt.polarities_dim)

    def forward(self, inputs, is_return_ensemble_values: bool = False):
        # get inputs
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        text_bert_indices_target_mask = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )

        # apply bert
        last_hidden_states = self.invoke_language_model(
            lm=self.language_model,
            input_ids=text_bert_indices,
        )

        # element-wise multiplication with target mask
        # unsqueeze, cf. https://stackoverflow.com/q/62559382
        text_bert_indices_target_mask_unsqueezed = text_bert_indices_target_mask.unsqueeze(
            -1
        )
        last_hidden_states_only_target = (
                last_hidden_states * text_bert_indices_target_mask_unsqueezed
        )

        # as in TD-BERT, perform max pooling
        last_hidden_states_only_target_aggregated, _ = last_hidden_states_only_target.max(
            dim=1
        )

        # dropout before dense layer, as in most other tsc models
        last_hidden_states_only_target_aggregated = self.dropout(
            last_hidden_states_only_target_aggregated
        )

        if is_return_ensemble_values:
            return last_hidden_states_only_target_aggregated
        else:
            # dense layer
            logits = self.fc(last_hidden_states_only_target_aggregated)
            # removed tanh, which was invoked in original tdbert. for training, we dont
            # need it to properly compute the loss. we would, however, need softmax during
            # inferring to have the probabilities of all mutually exclusive classes
            # to sum up to 1
            return logits



