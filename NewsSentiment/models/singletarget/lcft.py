# adapted from https://github.com/yangheng95/LCF-ABSA
from argparse import Namespace
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# from transformers.modeling_bert import BertPooler, BertSelfAttention

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class PointwiseFeedForward(nn.Module):
    """
    A two-feed-forward-layer module
    """

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = None  # BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), dtype=np.float32),
            dtype=torch.float32,
        ).to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCFT_BERT(FXBaseModel):
    @staticmethod
    def get_language_models():
        return (BERT_BASE_UNCASED,)

    @staticmethod
    def get_input_field_ids():
        return [
            (BERT_BASE_UNCASED, FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (
                BERT_BASE_UNCASED,
                FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
            ),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS),
            (BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            (BERT_BASE_UNCASED, FIELD_SYNTAX_DEPENDENCY_MATRIX),
        ]

    def __init__(self, transformer_models: Dict, opt: Namespace):
        super(LCFT_BERT, self).__init__()

        bert = transformer_models[BERT_BASE_UNCASED]
        self.bert_spc = bert
        self.opt = opt

        # self.bert_local = copy.deepcopy(bert)  # Uncomment the line to use dual Bert
        self.bert_local = (
            bert  # Default to use single Bert and reduce memory requirements
        )
        self.dropout = nn.Dropout(self.opt.dropout)
        self.bert_SA = SelfAttention(bert.config, self.opt)

        self.dependency_tree_to_attention_vector = nn.Linear(opt.max_seq_len, 1)
        # perform the softmax along each batch (dim 0)
        self.softmax = nn.Softmax(dim=0)

        # self.linear_double = nn.Linear(
        #    bert.config.hidden_size * 2, bert.config.hidden_size
        # )
        self.mean_pooling_double = PointwiseFeedForward(
            bert.config.hidden_size * 2,
            bert.config.hidden_size,
            bert.config.hidden_size,
        )
        self.linear_single = nn.Linear(bert.config.hidden_size, bert.config.hidden_size)
        self.bert_pooler = None  # BertPooler(bert.config)

        self.dense = nn.Linear(bert.config.hidden_size, self.opt.polarities_dim)

    def feature_dynamic_mask(self, text_local_indices, aspect_indices, distances_input):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        mask_len = self.opt.SRD
        masked_text_raw_indices = np.ones(
            (
                text_local_indices.size(0),
                self.opt.max_seq_len,
                self.bert_local.config.hidden_size,
            ),
            dtype=np.float32,
        )
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                # this should never be reached
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
                        (self.bert_local.config.hidden_size), dtype=np.float
                    )
                for j in range(asp_begin + asp_len + mask_len, self.opt.max_seq_len):
                    masked_text_raw_indices[text_i][j] = np.zeros(
                        (self.bert_local.config.hidden_size), dtype=np.float
                    )
                else:
                    distances_i = distances_input[text_i]
                for i, dist in enumerate(distances_i):
                    # iterate the distances and set the mask to 0 for those that have a
                    # too large distance
                    if dist > mask_len:
                        masked_text_raw_indices[text_i][i] = np.zeros(
                            (self.bert_local.config.hidden_size), dtype=np.float
                        )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def feature_dynamic_weighted(
        self, text_local_indices, aspect_indices, distances_input
    ):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones(
            (
                text_local_indices.size(0),
                self.opt.max_seq_len,
                self.bert_local.config.hidden_size,
            ),
            dtype=np.float32,
        )
        mask_len = self.opt.SRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
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
            else:
                distances_i = distances_input[text_i]  # distances of batch i-th
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(
                            texts[text_i]
                        )
                    else:
                        distances_i[i] = 1

                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = (
                        masked_text_raw_indices[text_i][i] * distances_i[i]
                    )
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs):
        text_target_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS,
        )
        text_target_bert_segments_ids = FXDataset.get_input_by_params(
            inputs,
            BERT_BASE_UNCASED,
            FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
        )
        text_local_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )
        aspect_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TARGET_IDS_WITH_SPECIAL_TOKENS
        )
        syntax_dependency_matrix = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_SYNTAX_DEPENDENCY_MATRIX
        )

        # apply bert
        bert_spc_out, _, _ = self.bert_spc(
            text_target_bert_indices, text_target_bert_segments_ids
        )
        bert_local_out, _, _ = self.bert_local(text_local_indices)

        # perform dependency weight vector weighting ("masking")
        dependency_weight_vector = self.dependency_tree_to_attention_vector(
            syntax_dependency_matrix
        )
        # we need to have some normalization because the other vectors that are stacked later are all yielded
        # by normalized weighting vectors. besides the implementation below, we for now simply use softmax, but
        # cf https://stats.stackexchange.com/questions/481798
        dependency_weight_vector_normalized = self.softmax(dependency_weight_vector)
        # since softmax sums to 1 in each batch item, we multiply this with the seqlen to match the "power" of the
        # other things that are later stacked (there, each weight scalar of the vector that is multiplied
        # with the bert output is between 0 and 1, whereas after softmax they all sum to 1)
        seqlen = text_target_bert_indices.shape[1]
        dependency_weight_vector_normalized = (
            dependency_weight_vector_normalized * seqlen
        )

        # we linearly normalize the dependency weight vector because its values will currently be just any float
        # we do this because the other (two) components that are concatenated later are either original bert's output
        # or bert's output multiplied with a weight mask (each scalar between 0 and 1), so we do the same here
        # shape: batch, seq, 1
        # for each batch item, it should be normalized independently of the other batch items so that each of scalar
        # in the batch item is between 0 and 1
        # perform sigmoid first to handle very large and very small values. after this values will be between 0 and 1
        # dependency_weight_vector = self.sigmoid(dependency_weight_vector)
        # dependency_weight_vector_min_batch_wise, _ = dependency_weight_vector.min(dim=1, keepdim=True)
        # dependency_weight_vector_max_batch_wise, _ = dependency_weight_vector.max(dim=1, keepdim=True)
        # actual linear normalization
        # dependency_weight_vector_normalized = dependency_weight_vector-dependency_weight_vector_min_batch_wise
        # dependency_weight_vector_normalized = dependency_weight_vector_normalized / (
        #        dependency_weight_vector_max_batch_wise - dependency_weight_vector_min_batch_wise
        # )
        # repeat for scalar wise multiplication
        dependency_weight_vector_normalized = dependency_weight_vector_normalized.repeat(
            1, 1, bert_local_out.shape[2]
        )

        # multiply with bert
        bert_local_out_weighted_dependency_tree = torch.mul(
            bert_local_out, dependency_weight_vector_normalized
        )

        out_cat = torch.cat(
            (bert_local_out_weighted_dependency_tree, bert_spc_out,), dim=-1,
        )
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out = self.bert_SA(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)

        dense_out = self.dense(pooled_out)
        return dense_out
