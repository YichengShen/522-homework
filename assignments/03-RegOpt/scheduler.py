import math
from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.

    This learning rate scheduler implements cosine annealing with warm restarts,
    which consists of cyclically varying the learning rate between
    a maximum and a minimum value using a cosine function,
    and periodically resetting the learning rate to its maximum value.

    The learning rate at each epoch is computed as:

        eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t / T))

    where `eta_min` is the minimum learning rate, `base_lr` is the initial learning rate,
    `t` is the current epoch within the current cycle, `T` is the length of the current cycle,
    and `pi` is the mathematical constant pi.

    The length of the current cycle is determined by
    `T_0` in the first cycle and `T_0` * `T_mult` in subsequent cycles.
    The minimum learning rate is determined by `eta_min`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of epochs in the first cycle.
        T_mult (float): Factor by which `T_0` is multiplied after each restart.
        eta_min (float): Minimum learning rate.
        last_epoch (int): The index of the last epoch.
    """

    def __init__(self, optimizer, T_0=1000, T_mult=2, eta_min=0.000006, last_epoch=-1):
        """
        Create a new scheduler.
        """

        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_T = T_0
        self.current_eta_min = eta_min
        self.end_of_cycle = False
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns a list of learning rates.
        """
        if self.last_epoch == 0:
            return self.base_lrs

        if self.end_of_cycle:
            self.current_T *= self.T_mult
            self.current_eta_min *= self.eta_min
            self.end_of_cycle = False
            self.last_epoch = -1

        return [
            self.current_eta_min
            + 0.5
            * (base_lr - self.current_eta_min)
            * (
                1
                + math.cos(
                    math.pi * (self.last_epoch % self.current_T) / self.current_T
                )
            )
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.end_of_cycle = self.last_epoch % self.current_T == self.current_T - 1
        super(CustomLRScheduler, self).step(epoch)


class CustomStepLRScheduler(_LRScheduler):
    """
    A custom learning rate scheduler.
    """

    def __init__(
        self,
        optimizer,
        decay_rate=0.999,
        decay_epochs=14,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Create a new scheduler.
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
