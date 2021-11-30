# adapted from ABSA-PyTorch
import torch
import torch.nn as nn

# CrossEntropyLoss for Label Smoothing Regularization
from NewsSentiment.SentimentClasses import SentimentClasses


class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, smoothing_value=0.2, weight=None):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.smoothing_value = smoothing_value
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.weight = weight

    def _create_smooth_one_hot_for_true_targets(
        self, labels, batch_size, num_classes, num_targets
    ):
        """
        Produces a smooth one hot encoded tensor where all "false" scalars are set to
        base_prob, and all "true" scalars set to base_prob + 1 - smoothing_value.
        Effectively, for example for three classes, this results in:
        False classes: 0.03 (0.3333*0.1=baseprob)
        True classes: 0.03 + 1 - 0.1 = 0.93
        :param labels: the true classes
        :param batch_size:
        :param num_classes:
        :return:
        """
        # prior label distribution is set uniform (see Sec 3.5,
        # https://arxiv.org/pdf/1902.09314.pdf )
        prior_distribution = 1.0 / float(num_classes)
        # calculate base probability
        base_prob = self.smoothing_value * prior_distribution
        # initialize 0 tensor
        one_hot_label = torch.zeros(
            batch_size, num_classes, num_targets, device=self.device
        )
        # set probability of all classes in all batches and targets to the base prob
        # (in normal one hot encoding, this would be 0 instead)
        one_hot_label = one_hot_label + base_prob

        if self.weight is not None:
            raise NotImplementedError(
                "test this first!!! currently untested with multi tsc"
            )
            one_hot_label = one_hot_label * self.weight

        # iterate over each single item of the batch
        for batch_index in range(batch_size):
            for target_index in range(num_targets):
                # get the class index
                class_index = labels[batch_index, target_index].item()
                if class_index == SentimentClasses.FILLUP_POLARITY_VALUE:
                    # cant set the one hot encoded here, since there is no "true" class
                    # need to ignore this later
                    pass
                else:
                    # set the class probability
                    one_hot_label[batch_index, class_index, target_index] += (
                        1.0 - self.smoothing_value
                    )

        return one_hot_label

    def forward(
        self, predictions: torch.Tensor, labels: torch.Tensor, size_average=True
    ):
        """
        Calculate k-dimensional cross entropy loss
        :param predictions: shape: (batch, polarities e.g. 3, targets e.g. 5)
        :param labels: shape: (batch, targets)
        :param size_average:
        :return:
        prediction size earlier was batch polarities
        """
        batch_size, num_classes, num_targets = predictions.size()
        # calculate log of probability of each class (of each batch and target)
        pre_logsoftmax = self.logSoftmax(predictions)

        smooth_one_hot_for_true_targets = self._create_smooth_one_hot_for_true_targets(
            labels, batch_size, num_classes, num_targets
        ).to(self.device)

        # ignore those comparison where labels has a fillup value (=to be ignored)
        mask_nonfillup = labels != SentimentClasses.FILLUP_POLARITY_VALUE
        # shape is batch, targets
        # shape should be batch, classes, targets
        mask_nonfillup = mask_nonfillup.unsqueeze(1).repeat(1, num_classes, 1)
        # convert to 1 for True and 0 for False
        mask_nonfillup = mask_nonfillup.to(torch.int)
        # multiply so that scalars to be ignored are set to 0 (resulting in 0 loss for
        # those scalars, i.e., targets)
        # notes: categorical cross entropy loss does not directly punish on a low level
        # those predictions (scalars) that belong to incorrect classes (defined by true
        # or here, labels) but is only calculated by comparing the one true class (
        # defined by true, or here, labels) where it has a 1 (one hot encoded). since
        # the probability is 100% of all classes (also the output of the neural network)
        # the loss still punishes wrong predictions, i.e., if the class probability
        # should be 100% but is only 25% or 70%, the loss will be non-zero)
        # as a consequence, when there is no right class in "true", there cannot be a
        # loss. so, the multiplication below, which sets all fillup-targets to 0, has
        # the expected effect (no loss can result from fill up values, as all their
        # class probabilities are set to 0)
        smooth_one_hot_for_true_targets = (
            smooth_one_hot_for_true_targets * mask_nonfillup
        )

        # multiply
        loss = -smooth_one_hot_for_true_targets * pre_logsoftmax

        # aggregate loss to scalar over classes
        loss = torch.sum(loss, dim=1)
        # aggregate loss to scalar over targets
        loss = torch.sum(loss, dim=1)

        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)
