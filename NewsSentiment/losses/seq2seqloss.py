import torch
import torch.nn as nn

from NewsSentiment.SentimentClasses import SentimentClasses


class SequenceLoss(nn.Module):
    """
    Input to this loss are sequences, see models/multitargets/seq2seq.py
    """

    def __init__(self, device, weight):
        super(SequenceLoss, self).__init__()
        self.device = device
        self.weight = weight
        self.actual_loss = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=SentimentClasses.FILLUP_POLARITY_VALUE,
        )

        assert self.weight is None, "not implemented, weight must be None"

    def forward(
        self,
        predictions: torch.Tensor,
        true_classes: torch.Tensor,
        true_target_mask: torch.Tensor,
    ):
        """
        :param predictions:         shape: batch, numclasses, seqlen
        :param true_classes:        shape: batch, targets
        :param true_target_mask:    shape: batch, targets, seqlen
        :return:
        """
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        seq_len = predictions.shape[2]

        # create true_sequence, shape: batch, seqlen
        true_sequence = (
            torch.ones([batch_size, seq_len], dtype=torch.int64, device=self.device)
            * SentimentClasses.FILLUP_POLARITY_VALUE
        )
        # update individual scalars
        for batch_index in range(batch_size):
            for target_index in range(true_target_mask.shape[1]):
                # get the sentiment class of this target in this batch
                true_sentiment_class = true_classes[batch_index, target_index].item()
                # if the true_sentiment_class is FILLUP_POLARITY_VALUE, ignore (no need
                # to update because we initialized the whole true_sequence with
                # FILLUP_POLARITY_VALUE
                if true_sentiment_class == SentimentClasses.FILLUP_POLARITY_VALUE:
                    continue

                # if not FILLUP_POLARITY_VALUE, update the true_sequence
                # iterate all tokens
                for seq_index in range(seq_len):
                    # determine if at the current token there is a target
                    is_target = true_target_mask[
                        batch_index, target_index, seq_index
                    ].item()

                    if is_target == 1:
                        # there is a target
                        # we should update the scalar in true_sequence at the
                        # corresponding part (thereby disregarding the target dimension,
                        # since all targets are merged into one dimension). ensure, that
                        # -100 is still there (if there is another value, this means
                        # that we have overlapping targets)
                        prev_value = true_sequence[batch_index, seq_index].item()
                        if prev_value == SentimentClasses.FILLUP_POLARITY_VALUE:
                            # the previous value is FILLUP_POLARITY_VALUE, so there is
                            # no target at this token already. so, we can update
                            true_sequence[batch_index, seq_index] = true_sentiment_class
                        else:
                            # there is already a target class -> overlapping targets
                            # this can happen for probably two reasons:
                            # 1) there are actually different targets in the data that
                            #    overlap
                            # 2) we duplicated a target in FXDataset
                            # either way, for now, if the value to be set is identical
                            # to the one already present, we continue, otherwise throw
                            # an error
                            if prev_value == true_sentiment_class:
                                pass
                            else:
                                raise ValueError(
                                    f"tried to update true_sequence[{batch_index},{seq_index}]={prev_value}"
                                )

                    elif is_target == 0:
                        # no target
                        # since we initialized the true_sequence tensor with -100
                        # scalars, there's not need to update the value for non-target
                        # nodes
                        pass
                    else:
                        raise ValueError(
                            f"true_target_mask must be either 0 or 1, is: {is_target}"
                        )

        loss = self.actual_loss(predictions, true_sequence)
        return loss
