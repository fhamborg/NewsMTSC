from argparse import Namespace
from typing import Dict

import torch.nn as nn

from NewsSentiment.SentimentClasses import SentimentClasses
from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset
from NewsSentiment.models.FXBaseModel import FXBaseModel


class SeqTwoSeqWithoutTargetMask(FXBaseModel):
    """
    Outputs the class probabilities for each token. So, the output will be:
    (batch, seqlen (150), classnum (3))
    """

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
        super(SeqTwoSeqWithoutTargetMask, self).__init__()
        self.language_model = transformer_models[BERT_BASE_UNCASED]
        self.dropout = nn.Dropout(opt.dropout)
        # https://pytorch.org/docs/stable/nn.html#linear supports multi-dimensional
        # input; only the last dimension has to be specified for Linear creation

        self.attentionlike_dense = nn.Linear(
            self.language_model.config.hidden_size,
            SentimentClasses.get_num_classes(),
        )

    def forward(self, inputs):
        # get inputs
        text_bert_indices = FXDataset.get_input_by_params(
            inputs, BERT_BASE_UNCASED, FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS
        )

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
        # shape: batch, seqlen, bertdim
        last_hidden_states = self.dropout(last_hidden_states)

        sequence_logits = self.attentionlike_dense(last_hidden_states)

        return sequence_logits
