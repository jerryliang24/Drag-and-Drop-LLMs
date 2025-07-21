from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class ModelInterface(ABC, Module):
    config = {}

    def __init__(self, config: dict):
        super().__init__()
        self.config.update(config)

    @abstractmethod
    def forward(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        """
        Forward and compute loss.
        :param source: Input without nan
        :param mask: Tensor.dtype = torch.bool, 1 for valid.
        :param condition: Tensor for condition.
        :param target: Target without nan
        :param kwargs: other parameters.
        :return: Tensor, one item loss.
        """

    @abstractmethod
    def generate(
        self, source: Tensor = None, mask: Tensor = None, condition: Tensor = None, target: Tensor = None, **kwargs
    ) -> Tensor:
        """
        Generate.
        :param source: Input without nan
        :param mask: Tensor.dtype = torch.bool, 1 for valid.
        :param condition: Tensor for condition.
        :param target: Should be kept to None.
        :param kwargs: other parameters.
        :return: Tensor, the same shape as target
        """

    @property
    def device(self):
        return next(self.parameters()).device
