"""
Define some functions to ensure compatability between vanilla RNNs, LSTMs and GRUs.
"""

# STD
from typing import Union, Tuple, Dict, Callable, Iterable, Any

# EXT
from torch import Tensor

# TYPES
AmbiguousHidden = Union[Tensor, Tuple[Tensor, Tensor]]


class RNNCompatabilityMixin:
    @staticmethod
    def hidden_compatible(hidden: AmbiguousHidden, func: Callable, *func_args: Tuple,
                          **func_kwargs: Dict) -> Any:
        """
        Ensure compatibility between GRU and LSTM RNNs by applying a function to both the hidden and cell state inside
        the hidden variable if necessary.
        """
        # LSTM case
        if type(hidden) == tuple:
            return func(hidden[0], *func_args, **func_kwargs), func(hidden[1], *func_args, **func_kwargs)
        # GRU / Vanilla RNN case
        else:
            return func(hidden, *func_args, **func_kwargs)

    @staticmethod
    def hidden_select(hidden: AmbiguousHidden) -> Tensor:
        """
        Ensure compatibility between GRU and LSTM RNNs by always selecting the hidden and not cell state inside
        the hidden variable if necessary.
        """
        # LSTM case
        if type(hidden) == tuple:
            return hidden[0]
        # GRU / Vanilla RNN case
        else:
            return hidden

    @staticmethod
    def hidden_scatter(hidden: AmbiguousHidden) -> Iterable[Iterable]:
        """
        Ensure compatibility between GRU and LSTM RNNs by always selecting returning hidden as an iterable.
        """
        # LSTM case
        if type(hidden) == tuple:
            return [hidden[0]], [hidden[1]]
        # GRU / Vanilla RNN case
        else:
            return [hidden]
