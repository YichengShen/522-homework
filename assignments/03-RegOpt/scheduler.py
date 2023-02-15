from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    """

    def __init__(
        self,
        optimizer,
        decay_rate=0.999,
        decay_epochs=200,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """

        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """
        Returns a list of learning rates.
        """

        if self.last_epoch % self.decay_epochs != 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.decay_rate for group in self.optimizer.param_groups]
