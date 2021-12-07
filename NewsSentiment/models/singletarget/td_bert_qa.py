# this file re-implements TD-BERT-QA by Gao Zhengjie et al.
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class TD_BERT_QA_MUL(FXBaseModel):
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
        super(TD_BERT_QA_MUL, self).__init__()
        self.opt = opt
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(self.language_model.config.hidden_size, opt.polarities_dim)  # 全连接层 bbfc
        self.bn = nn.BatchNorm1d(self.language_model.config.hidden_size)

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

        # as in TD-BERT, perform max pooling
        last_hidden_states_only_target_aggregated, _ = last_hidden_states_only_target.max(
            dim=1
        )

        target_in_sent_embed = self.bn(last_hidden_states_only_target_aggregated)
        target_in_sent_embed = target_in_sent_embed.mul(pooler_output)
        cat = self.dropout(target_in_sent_embed)

        logits = self.fc(cat)
        # removed tanh, which was invoked in original tdbert. for training, we dont
        # need it to properly compute the loss. we would, however, need softmax during
        # inferring to have the probabilities of all mutually exclusive classes
        # to sum up to 1

        return logits


class TD_BERT_QA_CON(FXBaseModel):
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
        super(TD_BERT_QA_CON, self).__init__()
        self.opt = opt
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(self.language_model.config.hidden_size*2, opt.polarities_dim)  # 全连接层 bbfc
        self.bn = nn.BatchNorm1d(self.language_model.config.hidden_size)

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

        # as in TD-BERT, perform max pooling
        last_hidden_states_only_target_aggregated, _ = last_hidden_states_only_target.max(
            dim=1
        )

        # not entirely sure whether this is as in original tdbertqa-con, because the code does not exist in the repo
        # (seems to be part of the commented lines in there)
        pooler_output = self.bn(pooler_output)

        cat = torch.cat([pooler_output, last_hidden_states_only_target_aggregated], dim=1)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        # removed tanh, which was invoked in original tdbert. for training, we dont
        # need it to properly compute the loss. we would, however, need softmax during
        # inferring to have the probabilities of all mutually exclusive classes
        # to sum up to 1

        return logits
