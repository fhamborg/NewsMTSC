import torch.nn as nn
import torch


class AggregatorForBert(nn.Module):
    """
    According to https://huggingface.co/transformers/model_doc/bert.html#bertmodel we
    should not (as in the original SPC version) the pooler_output but get the
    last_hidden_state and "averaging or pooling the sequence of hidden-states for the
    whole input sequence"
    """
    def __init__(self, spc_lm_representation: str):
        super(AggregatorForBert, self).__init__()
        self.spc_lm_representation = spc_lm_representation

    def forward(self, last_hidden_state, pooler_output, all_hidden_states):
        if self.spc_lm_representation == "pooler_output":
            prepared_output = pooler_output
        elif self.spc_lm_representation == "mean_last":
            prepared_output = last_hidden_state.mean(dim=1)
        elif self.spc_lm_representation == "mean_last_four":
            prepared_output = (
                torch.stack(all_hidden_states[-4:]).mean(dim=0).mean(dim=1)
            )
        elif self.spc_lm_representation == "mean_last_two":
            prepared_output = (
                torch.stack(all_hidden_states[-2:]).mean(dim=0).mean(dim=1)
            )
        elif self.spc_lm_representation == "mean_all":
            prepared_output = torch.stack(all_hidden_states).mean(dim=0).mean(dim=1)
        elif self.spc_lm_representation == "sum_last":
            prepared_output = last_hidden_state.sum(dim=1)
        elif self.spc_lm_representation == "sum_last_four":
            prepared_output = torch.stack(all_hidden_states[-4:]).sum(dim=0).sum(dim=1)
        elif self.spc_lm_representation == "sum_last_two":
            prepared_output = torch.stack(all_hidden_states[-2:]).sum(dim=0).sum(dim=1)
        elif self.spc_lm_representation == "sum_all":
            prepared_output = torch.stack(all_hidden_states).sum(dim=0).sum(dim=1)
        return prepared_output
