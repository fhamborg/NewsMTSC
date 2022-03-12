# adapted from https://github.com/yangheng95/LC-ABSA/blob/c945a94e0f86116c5578245aa9ad36c46c7b9c4a/models/lc_apc/lcf_bert.py
# according to
import copy
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
# from transformers.modeling_bert import BertPooler, BertSelfAttention

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.layers.attention import FXBertSelfAttention
from NewsSentiment.models.FXBaseModel import FXBaseModel


class GlobalContext(nn.Module):
    def __init__(self, global_context_seqs_per_doc):
        super(GlobalContext, self).__init__()
        self.global_context_seqs_per_doc = global_context_seqs_per_doc

    def forward(self, inputs):
        pass


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = FXBertSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_probs_dropout_prob=0.1,
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), dtype=np.float32),
            dtype=torch.float32,
        ).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_BERT2Dual(FXBaseModel):
    """
    While lcf.py:LCF_BERT is the implementation as implemented in PyTorch-ABSA repository, this implementation here
    (LCF_BERT2Dual) is following the implementation as in the author's repository, which according to
    https://github.com/yangheng95/LC-ABSA/issues/10#issuecomment-670301603 has seen some more improvements compared to
    the version from PyTorch-ABSA
    """

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
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (get_default_lm(), FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(LCF_BERT2Dual, self).__init__()

        bert = transformer_models[get_default_lm()]

        self.bert4global = bert
        # note that we use a second bert here, which should slightly improve performance
        # cf. https://github.com/yangheng95/LC-ABSA/#tips
        # self.bert4local = copy.deepcopy(bert)
        # we can't do this on scc because even for batch size = only 16 we run out of
        # memory. because of that, we use the same bert for both local and global
        # (just as in lcf.py)
        self.bert4local = bert
        self.opt = opt
        self.dropout = nn.Dropout(self.opt.dropout)
        self.bert_SA = SelfAttention(bert.config, self.opt)
        self.linear2 = nn.Linear(bert.config.hidden_size * 2, bert.config.hidden_size)
        # self.linear3 = nn.Linear(bert.config.hidden_size * 3, bert.config.hidden_size)
        self.bert_pooler = None # BertPooler(bert.config)
        self.dense = nn.Linear(bert.config.hidden_size, self.opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones(
            (
                text_local_indices.size(0),
                self.opt.max_seq_len,
                self.bert4local.config.hidden_size,
            ),
            dtype=np.float32,
        )
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
            except:
                continue
            if asp_begin >= mask_len:
                mask_begin = asp_begin - mask_len
            else:
                mask_begin = 0
            for i in range(mask_begin):
                masked_text_raw_indices[text_i][i] = np.zeros(
                    (self.bert4local.config.hidden_size), dtype=np.float
                )
            for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                masked_text_raw_indices[text_i][j] = np.zeros(
                    (self.bert4local.config.hidden_size), dtype=np.float
                )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones(
            (
                text_local_indices.size(0),
                self.opt.max_seq_len,
                self.bert4local.config.hidden_size,
            ),
            dtype=np.float32,
        )
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    distances[i] = 1 - (
                        abs(i - asp_avg_index) + asp_len / 2 - self.opt.SRD
                    ) / np.count_nonzero(texts[text_i])
                else:
                    distances[i] = 1
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = (
                    masked_text_raw_indices[text_i][i] * distances[i]
                )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_target_bert_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS,
        )
        text_target_bert_segments_ids = FXDataset.get_input_by_params(
            inputs,
            get_default_lm(),
            FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
        )
        text_local_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        aspect_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS
        )

        # bert
        global_context_features = self.invoke_language_model(
            self.bert4global,
            input_ids=text_target_bert_indices,
            token_type_ids=text_target_bert_segments_ids,
        )
        local_context_features = self.invoke_language_model(
            self.bert4local, text_local_indices
        )

        # mask
        if self.opt.local_context_focus == "cdm":
            lcf_matrix = self.feature_dynamic_mask(text_local_indices, aspect_indices)
        elif self.opt.local_context_focus == "cdw":
            lcf_matrix = self.feature_dynamic_weighted(
                text_local_indices, aspect_indices
            )

        # LCF layer
        lcf_features = torch.mul(local_context_features, lcf_matrix)
        lcf_features = self.bert_SA(lcf_features)

        cat_features = torch.cat((lcf_features, global_context_features), dim=-1)
        cat_features = self.linear2(cat_features)
        cat_features = self.dropout(cat_features)

        pooled_out = self.bert_pooler(cat_features)
        dense_out = self.dense(pooled_out)

        return dense_out
