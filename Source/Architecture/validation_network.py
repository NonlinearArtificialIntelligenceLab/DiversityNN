"""
Standard Pytorch implementation of the classification network to validate the activation functions
learned through the meta-learning process on different datasets and architectures.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Activation.InterpolantActivation import InterpolantActivation


class Network(ABC, nn.Module):
    """
    Structure for Neural Network
    """

    def __init__(self, options):
        super().__init__()
        """Class initializer
        Args:
            options [Options]: Parameters for the model
        """
        self.device = options.device
        self.dataset = options.dataset
        self.dataset_name = options.dataset_name
        self.batch_size = options.inner_batch_size
        self.seed = options.seed
        self.validation_mode = options.validation_mode # selects which activation(s) to use from the activation file
        self.act = InterpolantActivation(
            options.act_path, self.device, options.seed, self.validation_mode
        )
        self.loss_function = nn.CrossEntropyLoss().to(self.device)

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Runs one iteration of the network"""
        pass

    @abstractmethod
    def do_train(
        self, batches, do_training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the neural net on batches of data passed into it
        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
            do_training (bool, True by default): flags whether the model is to be run in
                                                training or evaluation mode
        Returns: total loss, total accuracy
        """
        pass

    def do_eval(
        self, batches, do_training: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convienience function for running do_train in evaluation mode
        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
            do_training (bool, False by default): flags whether the model is to be run in
                                                training or evaluation mode
        Returns: total loss, total accuracy
        """
        return self.do_train(batches, do_training=False)

    def get_model(self):
        """ " getter function to help easy storage of the model
        Args:
            None
        Returns: the model and its optimizer
        """
        return self, self.optimizer


class FCN(Network):
    """Fully connected network inheriting from network abstract class""",

    def __init__(self, options):
        super().__init__(options)
        self.fc1 = torch.nn.Linear(
            in_features=options.dataset.input_size,
            out_features=options.inner_hidden_size,
            bias=True,
        ).to(self.device)
        self.fc2 = torch.nn.Linear(
            in_features=options.inner_hidden_size,
            out_features=options.dataset.output_size,
            bias=True,
        ).to(self.device)
        self.sequence = torch.nn.Sequential(self.fc1, self.act, self.fc2)
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=options.inner_lr, momentum=options.inner_mu
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run one iteration of the feed forward Neural Network
        Args:
            data (torch.Tensor): input data passed into the network
        Returns:
            torch.Tensor: network output
        """
        return self.sequence(data)

    def do_train(
        self, batches, do_training: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the neural net on batches of data passed into it
        Args:
            batches (torch.dataset object): Shuffled samples of data for evaluation by the model
            do_training (bool, True by default): flags whether the model is to be run in
                                                training or evaluation mode
        Returns: total loss, total accuracy
        """
        if do_training:
            self.train()
        else:
            self.eval()
        total_loss = torch.zeros(1, device=self.device)
        total_acc = torch.zeros(1, device=self.device)
        for _, (data, target) in enumerate(batches):
            target = target.to(self.device)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = None
            data = self.dataset.fc_reshape(data).to(self.device)
            output = self.forward(data).to(self.device)
            loss = self.loss_function(output, target)

            if do_training is True:
                loss.backward()
                self.optimizer.step()

            predicted = output.argmax(dim=1, keepdim=True)
            accuracy = predicted.eq(
                target.view_as(predicted)
            ).sum().item() / target.size(0)
            total_loss += loss
            total_acc += accuracy
        total_loss = total_loss / len(batches.dataset) * self.batch_size
        total_acc = total_acc / len(batches.dataset) * self.batch_size
        return total_loss.cpu().detach(), total_acc.cpu().detach()
