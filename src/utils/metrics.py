"""
This module provides functionalities to set up evaluation metrics used in a neural network evaluation process.

Functions:
    get_metrics: Generates a collection of essential evaluation metrics.
"""
from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    """Generate a collection of essential evaluation metrics.

    This function creates a collection of metrics including F1 Score, Precision, and Recall using
    the provided keyword arguments for their initialization.

    Args:
        kwargs: Arbitrary keyword arguments that are forwarded to the initialization of each metric.

    Returns:
        MetricCollection: A collection of initialized metrics.
    """
    return MetricCollection(
        {
            "f1": F1Score(**kwargs),
            "precision": Precision(**kwargs),
            "recall": Recall(**kwargs),
        },
    )
