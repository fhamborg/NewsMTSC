import torch.nn as nn
import torch


class CrossEntropyLossWithConfidence(nn.Module):
    def __init__(self, weight, ignore_index):
        super(CrossEntropyLossWithConfidence, self).__init__()
        self.crossentropyloss = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index
        )
        self.loss_for_confidence = nn.HuberLoss()
        self.w_classes = 0.5

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor):
        # get the prediction components
        class_preds = predictions[:, 0:3, :]
        confidence_preds = predictions[:, 3:4, :]

        # calc the regular class-based loss
        class_loss = self.crossentropyloss(class_preds, labels)

        # now calc the confidence-based loss
        confidence_loss = 0

        # get the predicted classes
        predicted_classes = class_preds.argmax(dim=1)
        # and compare with the correct classes
        is_correct = torch.eq(predicted_classes, labels)
        is_correct_as_float = is_correct.float()
        # calc the confidence loss
        confidence_loss = self.loss_for_confidence(
            confidence_preds, is_correct_as_float
        )

        # calc the total loss
        total_loss = (
            self.w_classes * class_loss + (1 - self.w_classes) * confidence_loss
        )

        return total_loss
