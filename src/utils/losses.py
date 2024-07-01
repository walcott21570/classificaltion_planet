"""
This module provides functionalities to load and manage losses used in a neural network training pipeline.

Functions:
    get_losses: Creates a list of Loss objects based on a given list of LossConfig configurations.
"""

from dataclasses import dataclass
from typing import List

from torch import nn

from src.settings.config import LossConfig
from src.utils.general import load_object


@dataclass
class Loss(object):
    """Data class to encapsulate information about a specific loss.

    Attributes:
        name (str): Name of the loss.
        weight (float): Weight given to the loss during training.
        loss (nn.Module): PyTorch loss module representing the specific loss function.
    """

    name: str
    weight: float
    loss: nn.Module


def get_losses(losses_cfg: List[LossConfig]) -> List[Loss]:
    """Generate a list of Loss objects based on provided configurations.

    This function uses configurations provided in the form of LossConfig objects
    to create and initialize corresponding PyTorch loss modules.

    Args:
        losses_cfg (List[LossConfig]): List of loss configurations.

    Returns:
        List[Loss]: List of initialized Loss objects.
    """
    return [
        Loss(
            name=loss_cfg.name,
            weight=loss_cfg.weight,
            loss=load_object(loss_cfg.loss_fn)(**loss_cfg.loss_kwargs),
        )
        for loss_cfg in losses_cfg
    ]
