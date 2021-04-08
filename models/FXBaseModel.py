import torch.nn as nn

from abc import abstractmethod

from transformers import XLNetModel, AlbertModel, BertModel, RobertaModel


class FXBaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def get_language_models():
        return

    @staticmethod
    @abstractmethod
    def get_input_field_ids():
        return

    def invoke_language_model(self, lm, input_ids, token_type_ids=None):
        type_lm = type(lm)
        if type_lm == XLNetModel:
            last_hidden_state, mems, all_hidden_states = lm(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
        elif type_lm in [AlbertModel, BertModel, RobertaModel]:
            if token_type_ids is None:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids,
                )
            else:
                last_hidden_state, pooler_output, hidden_states = lm(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
        else:
            raise NotImplementedError

        return last_hidden_state
