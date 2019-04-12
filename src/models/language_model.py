"""
Implementation of a simple RNN language model.
"""

# STD
from typing import Optional, Dict, Tuple

# EXT
import torch
from torch import nn, Tensor
from overrides import overrides
from torch.autograd import Variable

# PROJECT
from src.models.abstract_rnn import AbstractRNN
from src.utils.types import HiddenDict, AmbiguousHidden, Device, StepSize
from src.utils.compatability import RNNCompatabilityMixin


# TODO: Debug Remove
def _mem_report(tensors, mem_type):
    '''Print the selected tensors of type
    There are two major storage types in our major concern:
        - GPU: tensors transferred to CUDA devices
        - CPU: tensors remaining on the system memory (usually unimportant)
    Args:
        - tensors: the tensors of specified type
        - mem_type: 'CPU' or 'GPU' in current implementation '''
    print('Storage on %s' % (mem_type))
    total_numel = 0
    total_mem = 0
    visited_data = []

    from collections import defaultdict
    tensor_count = defaultdict(int)

    for tensor in tensors:
        if tensor.is_sparse:
            continue
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.append(data_ptr)

        numel = tensor.storage().size()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
        total_mem += mem
        element_type = type(tensor).__name__
        size = tuple(tensor.size())

        tensor_count['%s\t%s' % (element_type, size)] += 1

        #print('%s\t%s\t%.2f' % (
        #    element_type,
        #    size,
        #    mem))
    print("Tensor\t(128, 650)", tensor_count["Tensor\t(128, 650)"])
    #print("\n".join(f"{tensor}: {count}" for tensor, count in tensor_count.items()))
    print(f"{len(tensor_count)} types of tensors found")
    print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem))


class LSTMLanguageModel(AbstractRNN):
    """
    Implementation of a LSTM language model that can process inputs token-wise or in sequences.
    """
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers, dropout, device: torch.device = "cpu"):
        """
        Parameters
        ----------
        vocab_size: int
            Size of input vocabulary.
        hidden_size: int
            Dimensionality of hidden activations.
        embedding_size: int
            Dimensionality of word embeddings.
        num_layers: int
            Number of RNN layers.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").
        """
        super().__init__("LSTM", hidden_size, embedding_size, num_layers, device)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.out_layer = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.dropout_layer = nn.Dropout(dropout)
        self.track_hidden_grad = True

        # Define parameters
        self.gates = {}
        self.decoder = nn.Linear(hidden_size, hidden_size)

        for l in range(num_layers):
            # Input to first layer is embedding, for others it's the hidden state of the previous layer
            input_size = embedding_size if l == 0 else hidden_size

            self.gates[l] = {
                'ii': nn.Linear(input_size, hidden_size),
                'if': nn.Linear(input_size, hidden_size),
                'ig': nn.Linear(input_size, hidden_size),
                'io': nn.Linear(input_size, hidden_size),
                'hi': nn.Linear(hidden_size, hidden_size),
                'hf': nn.Linear(hidden_size, hidden_size),
                'hg': nn.Linear(hidden_size, hidden_size),
                'ho': nn.Linear(hidden_size, hidden_size),
            }

            # Add gates to modules so that their parameters are registered by PyTorch
            for gate_name, gate in self.gates[l].items():
                super().add_module(f"{gate_name}_l{l}", gate)

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[HiddenDict] = None,
                **additional: Dict) -> Tuple[Tensor, HiddenDict]:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        device = self.current_device(reference=input_var)

        if hidden is None:
            batch_size = input_var.shape[0]
            hidden = {l: self.map(self.init_hidden(batch_size, device), self.track_grad) for l in range(self.num_layers)}

        # This is necessary when training on multiple GPUs - the batch of hidden states is moved back to main GPU
        # after every step
        else:
            hidden = {l: (h[0].to(device), h[1].to(device)) for l, h in hidden.items()}

        embed = self.embeddings(input_var)  # batch_size x seq_len x embedding_dim+
        embed = self.dropout_layer(embed)

        input_ = embed.squeeze(1)
        for l in range(self.num_layers):
            new_hidden = self.forward_step(l, hidden[l], input_)
            input_ = new_hidden[0]  # New hidden state becomes input for next layer
            hidden[l] = new_hidden  # Store for next step

        out = self.decoder(input_)
        out = self.dropout_layer(out)

        out = out.unsqueeze(1)
        output = self.predict_distribution(out)

        return output, hidden

    def forward_step(self, layer: int, hidden: AmbiguousHidden, input_: Tensor) -> AmbiguousHidden:
        """
        Do a single step for a single layer inside an LSTM.

        Parameters
        ----------
        layer: int
            Current layer number.
        hidden: AmbiguousHidden
            Tuple of hidden and cell state from the previous time step.
        input_: Tensor
            Input to the current layer: Either embedding if layer = 0 or hidden state from previous layer.

        Returns
        -------
        hx, cx: AmbiguousHidden
            New hidden and cell state for this layer.
        """
        hx, cx = hidden

        # Track gradients for these when doing recoding
        #if self.track_hidden_grad:
        #    hx = self.track_grad(hx)
        #    cx = self.track_grad(cx)

            # TODO: Remove debug
        #    import gc
        #    print(f"++++ Mem report post track vars  ++++")
        #    _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        # TODO: Employ PyTorch optimization with concatenated matrices?

        # Forget gate
        f_g = torch.sigmoid(self.gates[layer]['if'](input_) + self.gates[layer]['hf'](hx))

        # Input gate
        i_g = torch.sigmoid(self.gates[layer]['ii'](input_) + self.gates[layer]['hi'](hx))

        # Output gate
        o_g = torch.sigmoid(self.gates[layer]['io'](input_) + self.gates[layer]['ho'](hx))

        # Intermediate cell state
        c_tilde_g = torch.tanh(self.gates[layer]['ig'](input_) + self.gates[layer]['hg'](hx))

        # New cell state
        cx = f_g * cx + i_g * c_tilde_g

        # New hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx

    def predict_distribution(self, output: Tensor, out_layer: Optional[nn.Module] = None):
        """
        Generate the output distribution using an affine transformation.

        Parameters
        ----------
        output: Tensor
            Decoded output Tensor of current time step.
        out_layer: nn.Module
            Layer used to transform the current output to the distribution.

        Returns
        -------
        output_dist: Tensor
            Unnormalized output distribution for current time step.
        """
        # Default to models own output layer
        out_layer = out_layer if out_layer is not None else self.out_layer

        batch_size, seq_len, hidden_size = output.size()
        output_dist = out_layer(output.view(batch_size * seq_len, hidden_size))
        output_dist = output_dist.view(batch_size, seq_len, self.vocab_size)

        return output_dist

    def track_grad(self, var: Tensor) -> Tensor:
        """
        Track the (recoding) gradient of a non-leaf variable.
        """
        return var


class UncertaintyLSTMLanguageModel(LSTMLanguageModel, RNNCompatabilityMixin):
    """
    A LSTM Language model with an uncertainty recoding mechanism applied to it. This class is defined explicitly because
    the usual decorator functionality of the uncertainty mechanism prevents pickling of the model.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, mechanism_class, mechanism_kwargs,
                 device: torch.device = "cpu"):
        super().__init__(vocab_size, embedding_size, hidden_size, num_layers, dropout, device)
        self.mechanism = mechanism_class(model=self, **mechanism_kwargs)
        self.track_hidden_grad = True

    @overrides
    def forward(self, input_var: Tensor, hidden: Optional[Tensor] = None, **additional: Dict) -> Tuple[Tensor, Tensor]:
        """
        Process a sequence of input variables.

        Parameters
        ----------
        input_var: Tensor
            Current input variable.
        hidden: Tensor
            Current hidden state.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Decoded output Tensor of current time step.
        hidden: Tensor
            Hidden state of current time step after recoding.
        """
        device = self.current_device(reference=input_var)
        target_idx = additional.get("target_idx", None)
        out, hidden = super().forward(input_var, hidden, **additional)

        # Estimate uncertainty of those same predictions
        delta = self.get_perplexity(out, target_idx)

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        new_out_dist, new_hidden = self.recode_activations(hidden, delta, device)

        return new_out_dist, new_hidden

    @staticmethod
    def get_perplexity(out: Tensor, target_idx: Optional[Tensor] = None) -> Tensor:
        out = out.squeeze(1)

        # If target indices are not given, just use most likely token
        if target_idx is None:
            target_idx = torch.argmax(out, dim=1, keepdim=True)
        else:
            target_idx = target_idx.unsqueeze(1)

        target_probs = torch.gather(out, 1, target_idx)
        target_probs = torch.sigmoid(target_probs)
        target_ppls = 2 ** (-target_probs.log2())

        return target_ppls

    def recode_activations(self, hidden: HiddenDict, delta: Tensor, device: Device) -> Tuple[Tensor, HiddenDict]:
        """
        Recode all activations stored in a HiddenDict based on an error signal delta.

        Parameters
        ----------
        hidden: HiddenDict
            Dictionary of all hidden (and cell states) of all network layers.
        delta: Tensor
            Current error signal that is used to calculate the gradients w.r.t. the hidden states.
        device: torch.device
            Torch device the model is being trained on (e.g. "cpu" or "cuda").

        Returns
        -------
        new_out_dist, new_hidden: Tuple[Tensor, HiddenDict]
            New re-decoded output distribution alongside all recoded hidden activations.
        """
        # TODO: Remove debug
        import gc
        print("++++ Mem report recoding start ++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        # Register gradient hooks
        for l, hid in hidden.items():
            for h in hid:
                self.register_grad_hook(h)

        #hidden = {l: [self.register_grad_hook(h) for h in hid] for l, hid in hidden.items()}

        # TODO: Remove debug
        print("++++ Mem report post hooks++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        # Calculate gradient of uncertainty w.r.t. hidden states and make step
        delta.backward(gradient=torch.ones(delta.shape).to(device))

        #self.compute_recoding_gradient(delta, device)

        # TODO: Remove debug
        print("++++ Mem report recoding grad ++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        new_hidden = {
            l: tuple([
                # Use the step predictor for the corresponding state and layer
                self.recode(h, step_size=0.5)
                for h in hid])  # Be LSTM / GRU agnostic
            for l, hid in hidden.items()
        }

        del hidden

        # TODO: Remove debug
        print("++++ Mem report post recoding step ++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        # Re-decode current output
        new_out_dist = self.redecode_output_dist(new_hidden)

        # TODO: Remove debug
        print("++++ Mem report post decoding ++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        return new_out_dist, new_hidden

    def recode(self, hidden: Tensor, step_size: StepSize) -> Tensor:
        """
        Perform a single recoding step on the current time step's hidden activations.

        Parameters
        ----------
        hidden: Tensor
            Current hidden state.
        step_size: StepSize
            Batch size x 1 tensor of predicted step sizes per batch instance or one single float for the whole batch.

        Returns
        -------
        hidden: Tensor
            Recoded activations.predictor_layers: Iterable[int]
            Layer sizes for MLP as some sort of iterable.
        """
        import gc
        # Correct any corruptions
        hidden.grad = self.replace_nans(hidden.grad)
        # del hidden.recoding_grad
        # TODO: Remove debug
        print("++++ Mem report Remove grad++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        # Perform recoding by doing a gradient decent step
        #with torch.no_grad():
        hidden.add_(-step_size * hidden.grad)
        hidden.detach_()
            #del hidden

        print("++++ Mem report inner update ++++")
        _mem_report([obj for obj in gc.get_objects() if torch.is_tensor(obj)], "CPU")

        return hidden

    def train(self, mode=True):
        super().train(mode)
        self.mechanism.train(mode)

    def eval(self):
        super().eval()
        self.mechanism.eval()

    @staticmethod
    def replace_nans(tensor: Tensor) -> Tensor:
        """
        Replace nans in a PyTorch tensor with zeros.

        Parameters
        ----------
        tensor: Tensor
            Input tensor.

        Returns
        -------
        tensor: Tensor
            Tensor with nan values replaced.
        """
        tensor[tensor != tensor] = 0  # Exploit the fact that nan != nan

        return tensor

    def redecode_output_dist(self, new_hidden: HiddenDict) -> Tensor:
        """
        Based on the recoded activations, also re-decode the output distribution.

        Parameters
        ----------
        new_hidden: HiddenDict
            Recoded hidden activations for all layers of the network.

        Returns
        -------
        new_out_dist: Tensor
            Re-decoded output distributions.
        """
        num_layers = len(new_hidden.keys())
        new_out = self.decoder(self.select(new_hidden[num_layers - 1]))  # Select topmost hidden activations
        new_out = self.dropout_layer(new_out)
        new_out = new_out.unsqueeze(1)
        new_out_dist = self.predict_distribution(new_out)

        return new_out_dist

    def track_grad(self, var: Tensor) -> Tensor:
        """
        Track the (recoding) gradient of a non-leaf variable.
        """
        return Variable(var, requires_grad=True)

    @staticmethod
    def register_grad_hook(var: Tensor) -> None:
        """
        Register a hook that assigns the (recoding) gradient to a special attribute of the variable.

        Parameters
        ----------
        var: Tensor
            Variable we register the hook for.
        """

        def hook(grad: Tensor):
            var.grad = grad

        var.register_hook(hook)

        return var
