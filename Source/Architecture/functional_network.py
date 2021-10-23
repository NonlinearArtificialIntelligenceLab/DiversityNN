"""
Differentable form of vanilla FCN model for gradient based metalearning.
inspired from https://colab.research.google.com/drive/1P0dWB6WeOyFUU_-uO71i6D1cz24B_n3l
"""
import torch
import sys

sys.path.append("../")
from Driver import options
from typing import Type

# returns a tuple (state, function) for a vanilla MLP model
def functional_mlp(
    Options: Type[options], network: str, mixed=False
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Functional form of vanilla MLP model
    Args:
      Options (options): options object containing hyperparameters
      network (str): select between (outer) activation network or (inner) classification network
    Returns:
      (state, function) tuple
    """
    if network == "outer":
        D, H, O = 1, Options.outer_hidden_size, 1
    if network == "inner":
        D, H, O = (
            Options.dataset.input_size,
            Options.inner_hidden_size,
            Options.dataset.output_size,
        )
    linear1, linear2 = torch.nn.Linear(D, H), torch.nn.Linear(
        H, O
    ) 
    params = [linear1.weight, linear1.bias, linear2.weight, linear2.bias]
    params = [torch.Tensor(p.reshape(-1)) for p in params]
    param_vec = torch.cat(params).to(
        Options.device
    )  # flattened vector of parameters
    if network == "inner":

        def forward_fn(
            x: torch.Tensor, param_vec: torch.Tensor, afuncs: list
        ) -> torch.Tensor:
            """
            Forward function for vanilla MLP model by splitting the layer to combine activation functions
            Args:
                x (torch.Tensor): input tensor
                param_vec (torch.Tensor): flattened vector of parameters
                afuncs (list): list of activation functions
            """
            pointer = 0
            W1 = param_vec[pointer : pointer + D * H].reshape(H, D)
            pointer += D * H
            b1 = param_vec[pointer : pointer + H].reshape(1, -1)
            pointer += H
            W2 = param_vec[pointer : pointer + H * O].reshape(O, H)
            pointer += O * H
            b2 = param_vec[pointer : pointer + O].reshape(1, -1)

            activity = (W1 @ x.t()).t() + b1  # batches x hidden_size
            split_idx = int(H / Options.n_activations)
            h = []
            for i in range(Options.n_activations):
                h.append(afuncs[i](activity[:, i * split_idx : (i + 1) * split_idx]))

            h = torch.column_stack(h)
            h = h + 0.1 * (2 * torch.rand(*b1.shape).to(b1.device) - 1)
            out = (W2 @ h.t()).t() + b2
            return out

        return (
            param_vec,
            forward_fn,
        )  

    elif network == "outer":

        def forward_fn(x: torch.Tensor, param_vec: torch.Tensor, afunc) -> torch.Tensor:
            """
            Forward function for vanilla MLP model by splitting the layer to combine activation functions
            Args:
                x (torch.Tensor): input tensor
                param_vec (torch.Tensor): flattened vector of parameters
                afunc : activation
            """
            pointer = 0
            W1 = param_vec[pointer : pointer + D * H].reshape(H, D)
            pointer += D * H
            b1 = param_vec[pointer : pointer + H].reshape(1, -1)
            pointer += H
            W2 = param_vec[pointer : pointer + H * O].reshape(O, H)
            pointer += O * H
            b2 = param_vec[pointer : pointer + O].reshape(1, -1)

            h = afunc((W1 @ x.t()).t() + b1)
            h = h + 0.1 * (
                2 * torch.rand(*b1.shape).to(b1.device) - 1
            )  # add noise to prevent overfitting
            logits = (W2 @ h.t()).t() + b2
            return logits

        return param_vec, forward_fn
