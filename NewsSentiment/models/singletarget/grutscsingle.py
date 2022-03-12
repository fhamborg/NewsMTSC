from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from NewsSentiment.consts import *
from NewsSentiment.dataset import FXDataset, FXEasyTokenizer
from NewsSentiment.models.FXBaseModel import (
    FXBaseModel,
    provide_pretrained,
    default_pretrained,
)


@default_pretrained("v1.0.0")
@provide_pretrained(
    "v1.0.0", "https://github.com/fhamborg/NewsMTSC/releases/download/v1.0.0/grutsc"
)
class GRUTSCSingle(FXBaseModel):
    """
    Inspired from https://arxiv.org/pdf/2006.00052.pdf
    Differences:
    - instead of question ("Is the ACLU good for USA?") then text (1 or more sentences),
      we use text then target (and no question, similar to BERT-SPC)
    - no vader
    - additionally we can flexibly use any knowledge source as well as multiple
    - we have one large matrix for all concatenated knowledge source embeddings, whereas
      in the original paper they use individual, smaller matrices for each knowledge
      source embedding
    - target mask (mostly useful for BERT)
    - fine-tuning LM enabled
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
            # while we use text-then-target as bert input, we can use text targetmask
            # and text knowledge source mask because it is identical to a hypothetical
            # text-then-target target mask or text-then-target knowledge source mask
            # (we would not highlight the target in the 2nd component in the
            # corresponding mask)
            (get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK),
            (
                get_default_lm(),
                FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES,
            ),
        ]

    def __init__(
        self, transformer_models: Dict, opt: Namespace, config: PretrainedConfig
    ):
        super().__init__(config)
        self.language_model = transformer_models[get_default_lm()]
        self.ks_embeddings_dense = nn.Linear(
            FXEasyTokenizer.NUM_CATEGORIES_OF_SELECTED_KNOWLEDGE_SOURCES,
            self.language_model.config.hidden_size,
        )
        if get_default_lm() == BERT_BASE_UNCASED:
            self.is_use_targetmask = True
        else:
            self.is_use_targetmask = False

        num_input_embeddings = 2
        if self.is_use_targetmask:
            num_input_embeddings = 3

        self.gru = nn.GRU(
            self.language_model.config.hidden_size * num_input_embeddings,
            self.language_model.config.hidden_size * num_input_embeddings,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(opt.dropout)
        num_output_dim = opt.polarities_dim
        if opt.is_return_confidence:
            num_output_dim += 1

        self.dense = nn.Linear(
            # 3 inputs (original last gru out, mean, max), 2 inputs to gru (bert and
            # knowledge embedding), 2 (because bidirectional gru)
            self.language_model.config.hidden_size * 3 * num_input_embeddings * 2,
            num_output_dim,
        )

    def forward(self, inputs, is_return_ensemble_values: bool = False):
        # get inputs
        text_target_bert_indices = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS
        )
        text_target_bert_segments_ids = FXDataset.get_input_by_params(
            inputs,
            get_default_lm(),
            FIELD_TEXT_THEN_TARGET_IDS_WITH_SPECIAL_TOKENS_SEGMENT_IDS,
        )
        text_bert_indices_target_mask = FXDataset.get_input_by_params(
            inputs, get_default_lm(), FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_TARGET_MASK
        )
        text_bert_indices_selected_knowledge_sources = FXDataset.get_input_by_params(
            inputs,
            get_default_lm(),
            FIELD_TEXT_IDS_WITH_SPECIAL_TOKENS_SELECTED_KNOWLEDGE_SOURCES,
        )

        # apply bert
        last_hidden_states = self.invoke_language_model(
            lm=self.language_model,
            input_ids=text_target_bert_indices,
            token_type_ids=text_target_bert_segments_ids,
        )
        # shape: batch, seqlen, hiddendim

        # apply knowledge embedding
        knowledge_embedded = self.ks_embeddings_dense(
            text_bert_indices_selected_knowledge_sources.float()
        )
        # shape: batch, seqlen, hiddendim

        if self.is_use_targetmask:
            # repeat
            target_mask = text_bert_indices_target_mask.unsqueeze(dim=2).repeat(
                1, 1, knowledge_embedded.shape[2]
            )
            # shape: batch, seqlen, hiddendim

            # concat (called x_t in paper)
            bert_and_knowledge = torch.cat(
                (last_hidden_states, knowledge_embedded, target_mask), dim=2
            )
            # batch x seq x bert+knowledge+targetmask
        else:
            # concat (called x_t in paper)
            bert_and_knowledge = torch.cat(
                (last_hidden_states, knowledge_embedded), dim=2
            )
            # batch x seq x bert+knowledge

        # apply gru (result called z_t in paper)
        gru_all_hidden, gru_last_hidden = self.gru(
            bert_and_knowledge,
            torch.zeros(
                2,
                bert_and_knowledge.shape[0],
                self.language_model.config.hidden_size * 2,
            ).to(self.device),
        )
        # all hidden shape: batch x seq x 4*hidden (contains hidden states for each
        # part of the input seq)
        # last hidden shap: numdir x batch x 2*hidden (contains hidden states for last
        # part of input seq)

        # gru_last_hidden_own = gru_all_hidden[:,-1:,]
        # get both directions
        gru_last_hidden_dir0 = gru_last_hidden[0, :, :]
        gru_last_hidden_dir1 = gru_last_hidden[1, :, :]
        # shape each: batch x 2*hidden
        gru_last_hidden_stacked = torch.cat(
            (gru_last_hidden_dir0, gru_last_hidden_dir1), dim=1
        )
        # batch x 4*hidden

        # pooling
        # according to original paper: "max-pooling returns a vector with maximum
        # weights across all hidden states of input tokens for each dimension. in this
        # way, the input tokens with higher weights will be engaged for stance
        # prediction."
        gru_avg = torch.mean(gru_all_hidden, dim=1)
        gru_max, _ = torch.max(gru_all_hidden, dim=1)

        # concat (called "u" in original paper)
        gru_complete_concatted = torch.cat(
            (gru_last_hidden_stacked, gru_avg, gru_max), dim=1
        )

        if is_return_ensemble_values:
            return gru_complete_concatted
        else:
            # dense
            logits = self.dense(gru_complete_concatted)

            return logits
