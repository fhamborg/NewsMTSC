import torch
import torch.nn as nn

# CrossEntropyLoss with additional cross weight loss for two targets
from NewsSentiment.SentimentClasses import SentimentClasses


class CrossEntropyLoss_CrossWeight(nn.Module):
    def __init__(self, device, ignore_index, weight=None, crossloss_weight=0.2):
        super(CrossEntropyLoss_CrossWeight, self).__init__()
        self.device = device
        self.class_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,)
        # cosine similarity by default
        self.cross_weight_loss = nn.CosineSimilarity(dim=2)
        self.crossloss_weight = crossloss_weight
        self.crossentropyloss_weight = 1.0 - crossloss_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        cross_weight: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        """
        Calculate k-dimensional cross entropy loss by comparing predictions and
        true_labels. Additionally, calculates a cross_weight loss that maximizes the
        differences between the weights compared between both targets.

        :param predictions: shape: (batch, polarities e.g. 3, targets ie 2)
        :param true_labels: shape: (batch, targets)
        :param cross_weight:shape: (batch, targets, seqlen, bertdim)
        :return:
        prediction size earlier was batch polarities
        """
        batch_size, num_classes, num_targets = predictions.size()
        bert_dim = cross_weight.shape[3]

        # calculate regular cross entropy loss
        classification_loss = self.class_loss(predictions, true_labels)

        # calculate cross-weight loss
        # we must not calc the cross weight loss for batch item of two targets,
        # where 1 or 2 is a fill up target, i.e., the true value is
        # SentimentClasses.FILLUP_POLARITY_VALUE
        # get all batch items where the first target is fill up
        is_ignored_a = true_labels[:, 0]
        is_ignored_a = is_ignored_a == SentimentClasses.FILLUP_POLARITY_VALUE
        # get all batch items where the second is
        is_ignored_b = true_labels[:, 1]
        is_ignored_b = is_ignored_b == SentimentClasses.FILLUP_POLARITY_VALUE
        # logical or both lists to one
        is_ignored_batch_item = is_ignored_a | is_ignored_b
        # select only those batch items where no target
        is_not_ignored_batch_item = ~is_ignored_batch_item
        target_mask = target_mask[is_not_ignored_batch_item, :, :]
        count_non_ignored_batch_items = target_mask.shape[0]

        # if we have identical targets (=target masks) set cross weight loss to 0
        target_mask_a = target_mask[:, 0, :]
        target_mask_b = target_mask[:, 1, :]
        diff_target_mask = target_mask_a - target_mask_b
        diff_target_mask = diff_target_mask.sum(dim=1)
        # shape: batch
        # diff target mask, will be 0 if the two targets in one batched item are
        # identical, and 1 if they are different
        is_different_target_per_two_batch_items = diff_target_mask != 0

        # only selecting different targets effectively zeros out values of identical
        # targets
        cross_weight = cross_weight[is_different_target_per_two_batch_items]

        # if there is not at least a single batch item with different targets, shape[0]
        # will be 0. we use this to test for this condition and if all targets over all
        # batch items are identical, we skip the cross weight part
        count_different_targets = cross_weight.shape[0]
        if count_different_targets == 0:
            cross_weight_loss = 0
        else:
            assert count_different_targets >= 1
            seq_len = cross_weight.shape[2]
            weight_a = cross_weight[:, 0, :, :]
            # weight_b = cross_weight[:, 1, :, :]
            weight_b = weight_a

            # we add the negative sign, since we actually want to maximize the distance
            # between both vectors
            cross_weight_similarity = self.cross_weight_loss(weight_a, weight_b)
            # cross_weight_similarity will be -1 for absolutely dissimilar values
            # 0 for unrelated and +1 for identical values
            # normalize between 0 and 1
            cross_weight_similarity = (cross_weight_similarity + 1) / 2
            # 0 = dissimilar, 1 = identical
            # mean over seq len
            cross_weight_loss = cross_weight_similarity.mean(dim=1)
            # at this point we have for each batch item its loss (0 if dissimilar
            # targets, 1 if identical target in the batch item)
            # finally, compute the single loss: sum
            cross_weight_loss = cross_weight_loss.sum()
            # normalize (divide by number of batch items; note that this can be different
            # from the number of different targets in the batch)
            cross_weight_loss = cross_weight_loss / count_non_ignored_batch_items

        # total_loss
        total_loss = (
            self.crossentropyloss_weight * classification_loss
            + self.crossloss_weight * cross_weight_loss
        )

        return total_loss
