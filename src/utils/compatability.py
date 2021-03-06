"""
Define some functions to ensure compatability between vanilla RNNs, LSTMs and GRUs.
"""

# STD
from typing import Callable, Iterable, Any

# EXT
from torch import Tensor

# PROJECT
from src.utils.types import AmbiguousHidden


class RNNCompatabilityMixin:
    """
    Define the following operations on hidden variables with a single hidden state or hidden / cell state tuple:

    map: Apply a function to every every state.
    select: Only select the actual hidden state, discard the rest.
    scatter: Wrap every state inside another iterable.
    """
    @staticmethod
    def map(hidden: AmbiguousHidden, func: Callable, *func_args: Any, **func_kwargs: Any) -> Any:
        """
        Ensure compatibility between GRU and LSTM RNNs by applying a function to both the hidden and cell state inside
        the hidden variable if necessary.

        Parameters
        ----------
        hidden: AmbiguousHidden
            Either one hidden state or tuple of hidden and cell state.
        func: Callable
            Function being applied to hidden activations.
        func_args: Any
            Unnamed additional arguments for function.
        func_kwargs: Any
            Named additional arguments for function.

        Returns
        -------
        result: Any
            Function return value.
        """
        # LSTM case
        if type(hidden) in (tuple, list):
            return func(hidden[0], *func_args, **func_kwargs), func(hidden[1], *func_args, **func_kwargs)
        # GRU / Vanilla RNN case
        else:
            return func(hidden, *func_args, **func_kwargs)

    @staticmethod
    def select(hidden: AmbiguousHidden) -> Tensor:
        """
        Ensure compatibility between GRU and LSTM RNNs by always selecting the hidden and not cell state inside
        the hidden variable if necessary.

        Parameters
        ----------
        hidden: AmbiguousHidden
            Either one hidden state or tuple of hidden and cell state.

        Returns
        -------
        hidden: Tensor:
            Only the hidden state.
        """
        # LSTM case
        if type(hidden) == tuple:
            return hidden[0]
        # GRU / Vanilla RNN case
        else:
            return hidden
