# adapted from absa-pytorch
from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from NewsSentiment.dataset import FXDataset
from NewsSentiment.layers.attention import Attention
from NewsSentiment.layers.point_wise_feed_forward import PositionwiseFeedForward
from NewsSentiment.layers.squeeze_embedding import SqueezeEmbedding

from NewsSentiment.consts import *
from NewsSentiment.fxlogger import get_logger
from NewsSentiment.models.FXBaseModel import FXBaseModel

logger = get_logger()


class AEN_Base(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (BERT_BASE_UNCASED,)

    @staticmethod
    def get_input_field_ids():
        return [
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(AEN_Base, self).__init__()
        logger.info("creating AEN_Base")
        self.device = opt.device

        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.name = "aen_bert"
        self.lm_representation = "last"
        embed_dim = self.language_model.config.hidden_size

        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        hidden_dim = embed_dim  # or should this be 300, as mentioned in the paper

        self.attn_k = Attention(
            embed_dim,
            out_dim=hidden_dim,
            n_head=8,
            score_function="mlp",
            dropout=opt.dropout,
        )
        self.attn_q = Attention(
            embed_dim,
            out_dim=hidden_dim,
            n_head=8,
            score_function="mlp",
            dropout=opt.dropout,
        )
        self.ffn_c = PositionwiseFeedForward(hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(
            hidden_dim, n_head=8, score_function="mlp", dropout=opt.dropout
        )

        self.dense = nn.Linear(hidden_dim * 3, opt.polarities_dim)

    def apply_lm(self, _input, _input_attention=None):
        if self.name in ["aen_bert", "aen_roberta"]:
            last_hidden, _, all_hidden = self.language_model(
                input_ids=_input, attention_mask=_input_attention
            )
        elif self.name == "aen_distilbert":
            last_hidden, all_hidden = self.language_model(
                input_ids=_input, attention_mask=_input_attention
            )
        else:
            raise Exception("unknown model name")

        if self.lm_representation == "last":
            return last_hidden
        elif self.lm_representation == "sum_last_four":
            last_four = all_hidden[-4:]  # list of four, each has shape: 16, 80, 768
            last_four_stacked = torch.stack(last_four)  # shape: 4, 16, 80, 768
            sum_last_four = torch.sum(last_four_stacked, dim=0)
            return sum_last_four
        elif self.lm_representation == "mean_last_four":
            last_four = all_hidden[-4:]  # list of four, each has shape: 16, 80, 768
            last_four_stacked = torch.stack(last_four)  # shape: 4, 16, 80, 768
            mean_last_four = torch.mean(last_four_stacked, dim=0)
            return mean_last_four
        elif self.lm_representation == "sum_last_two":
            last_two = all_hidden[-2:]
            last_two_stacked = torch.stack(last_two)
            sum_last_two = torch.sum(last_two_stacked, dim=0)
            return sum_last_two
        elif self.lm_representation == "mean_last_two":
            last_two = all_hidden[-2:]
            last_two_stacked = torch.stack(last_two)
            mean_last_two = torch.mean(last_two_stacked, dim=0)
            return mean_last_two
        elif self.lm_representation == "sum_all":
            all_stacked = torch.stack(all_hidden)
            sum_all = torch.sum(all_stacked, dim=0)
            return sum_all
        elif self.lm_representation == "mean_all":
            all_stacked = torch.stack(all_hidden)
            mean_all = torch.mean(all_stacked, dim=0)
            return mean_all

    def forward(self, inputs):
        context = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        target = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS
        )
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)

        context = self.squeeze_embedding(context, context_len)
        # context_attention = self.squeeze_embedding(context_attention, context_len)
        context = self.apply_lm(context)
        context = self.dropout(context)

        target = self.squeeze_embedding(target, target_len)
        # target_attention = self.squeeze_embedding(target_attention, target_len)
        target = self.apply_lm(target)
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)

        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.device)

        hc_mean = torch.div(
            torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1)
        )
        ht_mean = torch.div(
            torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1)
        )
        s1_mean = torch.div(
            torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1)
        )

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)

        return out
