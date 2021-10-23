from scipy import interpolate
import numpy as np
import torch
import torch.nn as nn
from .interp1d import Interp1d


def func(x, target):
    """
    takes the interpolation function and creates a function that can be used as an activation function
    """
    interpolation = Interp1d()

    def f(xnew):
        return interpolation(x, target, xnew)

    return f


class InterpolantActivation(nn.Module):
    """Custom activation function from interpolation of file input
    Args:
        PATH (str): PATH to text array of activations

    """

    def __init__(self, PATH: str, device: str, seed: int, validation_mode: int = False):
        super().__init__()
        self.PATH = PATH
        self.act_array = torch.tensor(
            np.loadtxt(PATH), dtype=torch.float, device=device
        )
        self.n_activations = self.act_array.shape[0] if self.act_array.ndim > 1 else 1
        self.device = device
        self.seed = seed
        x = torch.linspace(-5, 5, 50).to(self.device)
        self.act_interp = []
        if self.n_activations == 1:
            act = func(x, self.act_array)
            self.act_interp.append(act)
        else:
            for i in range(self.n_activations):
                act = func(x, self.act_array[i])
                self.act_interp.append(act)
        if validation_mode:
            self.n_activations = 1
            self.act_interp = [func(x, self.act_array[validation_mode - 1])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the activation function
        Args:
            x ([FloatTensor]): input tensor
        Returns:
            [FloatTensor]: output tensor
        """
        layer_size = x.size(1)
        if self.n_activations == 1:
            return self.act_interp[0](x)
        else:
            split_x = int(layer_size // self.n_activations)
            beta = torch.arange(layer_size, device=self.device)
            result = torch.zeros(layer_size, device=self.device)
            for i, act in enumerate(self.act_interp):
                beta_mask = torch.logical_and(
                    i * split_x <= beta, beta < (i + 1) * split_x
                )
                component = act(x * (beta_mask).int().float())
                result = component if result is None else result + component
            return result.to(self.device)
