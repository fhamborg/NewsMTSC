# taken from https://github.com/Bjarten/early-stopping-pytorch
# Copyright by Bjarten
# License: MIT

from NewsSentiment.fxlogger import get_logger


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=2, delta=0.01):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 2
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.logger = get_logger()
        self.flag_has_score_increased_since_last_check = False

    def __call__(self, dev_score):
        if self.best_score is None:
            self.best_score = dev_score
            self.logger.debug(f'computed first dev-set score {dev_score:.6f}).')
            self.flag_has_score_increased_since_last_check = True
        elif dev_score < self.best_score + self.delta:
            self.counter += 1
            self.logger.debug(
                f'patience counter: {self.counter} out of {self.patience} (cur-score: {dev_score}, best-score:'
                f' {self.best_score})')
            self.flag_has_score_increased_since_last_check = False
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = dev_score
            self.counter = 0
            self.logger.debug(f'dev-set score increased ({self.best_score:.6f} --> {dev_score:.6f}).')
            self.flag_has_score_increased_since_last_check = True
