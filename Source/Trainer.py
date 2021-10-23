"""
Trainer for gradient based meta-learning.
"""
import torch
from dataclasses import dataclass, field
import numpy as np
from typing import Type
from Architecture.functional_network import functional_mlp
from Driver import options


class Agent:
    """Driver class for diversity network
    Attributes:
        options (dict): configuration for the nn
    Methods:
        accuracy (float): computes the accuracy of the model

    """

    def __init__(self, Options: Type[options]) -> None:
        """Sets up a new agent
        Args:
            Options (options): configuration options
        Returns:
            None
        """

        self.options = Options
        self.device = Options.device
        self.dataset = Options.dataset
        self.train_loader, self.test_loader = self.dataset.process()
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.options.act_path is None:
            if Options.activation_name == "tanh":
                self.base_activation = torch.tanh
            elif Options.activation_name == "sin":
                self.base_activation = torch.sin
            elif Options.activation_name == "elu":
                self.base_activation = torch.nn.ELU()
            self.n_activations = Options.n_activations

        @dataclass
        class innerResults:
            """
            Store the results of the inner optimization
            """

            train_loss: torch.Tensor = torch.zeros(
                self.options.n_epochs * len(self.train_loader), device=self.device
            )
            test_acc: torch.Tensor = torch.zeros(
                self.options.n_epochs, device=self.device
            )

        self.inner_results = innerResults()

    def accuracy(self, pred: torch.Tensor, targets: torch.Tensor) -> float:
        """Computes the accuracy of the model
        Args:
            pred (torch.Tensor): predictions
            targets (torch.Tensor): targets
        Returns:
            float: accuracy
        """
        return 100 * sum(torch.max(pred, 1)[1] == targets) / len(targets)

    def inner_optimization(self, model, afuncs: list):
        """Performs inner optimization
        Args:
            model (torch.nn.Module): model to optimize
            afuncs (list): activation functions
        Returns:
            inner_results (innerResults): result of the inner optimization
        """
        (param_vec, fwd_fn) = model

        avg_sq_grad = torch.ones_like(
            param_vec
        )  # set inital gradients to 1 for RMSProp
        counter = 0
        for epoch in range(self.options.n_epochs):
            for batch_idx, (inp, target) in enumerate(self.train_loader):
                inp, target = inp.to(self.device), target.to(self.device)
                loss = self.criterion(fwd_fn(inp, param_vec, afuncs), target)
                grads = torch.autograd.grad(
                    loss, param_vec, retain_graph=True, create_graph=True
                )[0]
                avg_sq_grad = avg_sq_grad * self.options.inner_mu + grads ** 2 * (
                    1 - self.options.inner_mu
                )
                param_vec = param_vec - self.options.inner_lr * grads / (
                    torch.sqrt(avg_sq_grad) + 1e-08
                )
                self.inner_results.train_loss[counter] = loss
                counter += 1

            for (inp, target) in self.test_loader:
                inp, target = inp.to(self.device), target.to(self.device)
                pred = fwd_fn(inp, param_vec, afuncs)
                test_acc = self.accuracy(pred, target)
            self.inner_results.test_acc[epoch] = test_acc
        return self.inner_results

    def mlp_afunc(self, model):
        """
        network that will behave as an activation function
        """
        (param_vec, fwd_fn) = model

        def activation(x: torch.Tensor) -> torch.Tensor:
            """
            Neural network activation function
            Args:
                x (torch.Tensor): input
            Returns:
                torch.Tensor: output
            """

            x_hat = 0.2 * fwd_fn(x.reshape(-1, 1), param_vec, torch.tanh)
            x_hat = x_hat.reshape(*x.shape)
            return self.base_activation(x) + x_hat

        return activation

    def outer_optimization(self):
        outer_param_vecs = []
        outer_fwd_fns = []
        outer_models = []
        outer_momenta = []
        outer_grads = []
        grad_norms = np.zeros(self.n_activations)
        inner_afuncs = []

        @dataclass
        class outerResults:
            inner_test_acc: list[np.ndarray] = field(default_factory=list)
            inner_afuncs: list[np.ndarray] = field(
                default_factory=list
            )  # activation output on linspace(-5, 5, 100)
            grad_norms: list[np.ndarray] = field(default_factory=list)
            train_losses: torch.Tensor = torch.zeros(
                self.options.steps, device=self.device
            )

        outer_results = outerResults()
        for i in range(self.n_activations):
            outer_models.append(functional_mlp(self.options, network="outer"))
            outer_param_vecs.append(outer_models[i][0])
            outer_fwd_fns.append(outer_models[i][1])
            outer_momenta.append(torch.zeros_like(outer_param_vecs[i]))
        with torch.autograd.set_detect_anomaly(True):
            for step in range(self.options.steps):
                # running inner optimization
                inner_afuncs = [
                    self.mlp_afunc(outer_models[i]) for i in range(self.n_activations)
                ]

                inner_model = functional_mlp(
                    self.options, network="inner"
                )  # randomly intializes inner MLP
                inner_results = self.inner_optimization(inner_model, inner_afuncs)
                outer_loss = inner_results.train_loss[-100:].mean()
                # running outer optimization

                for i in range(self.n_activations):
                    (outer_param_vecs[i], outer_fwd_fns[i]) = outer_models[i]
                # calculating meta-gradients
                outer_grads = [
                    *torch.autograd.grad(outer_loss, inputs=outer_param_vecs)
                ]

                for i in range(self.n_activations):
                    outer_grads[i] = outer_grads[i].clamp(min=-1e-01, max=1e-01)
                    outer_momenta[i] = (
                        self.options.outer_mu * outer_momenta[i] + outer_grads[i]
                    )
                    outer_param_vecs[i] = (
                        outer_param_vecs[i] - self.options.outer_lr * outer_momenta[i]
                    )
                    outer_models[i] = (outer_param_vecs[i], outer_fwd_fns[i])
                    grad_norms[i] = outer_grads[i] @ outer_grads[i]

                outer_results.grad_norms.append(grad_norms)

                # recording results
                outer_results.train_losses[step] = outer_loss.item()
                if step % 5 == 0 or step == self.options.steps - 1:
                    print(
                        "Step: {}, meta_gradient norms:{}, mean_inner_loss:{}, final_inner_loss:{}, test_accuracy:{}".format(
                            step,
                            outer_results.grad_norms[step],
                            outer_results.train_losses[step],
                            inner_results.train_loss[-1],
                            inner_results.test_acc[-1],
                        )
                    )

                # sampling activation function for plotting
                x_fn = torch.linspace(-5, 5, 100).to(self.device)
                outer_results.inner_afuncs = [
                    inner_afuncs[i](x_fn).clone().detach().cpu().numpy()
                    for i in range(self.n_activations)
                ]
                outer_results.inner_test_acc.append(
                    inner_results.test_acc.detach().cpu().numpy()
                )
        return outer_results
