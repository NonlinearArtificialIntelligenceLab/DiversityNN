"""
Trainer for using the standard pytorch model for validating the metalearned activations.
"""
import torch
import numpy as np
from typing import Union, Tuple
from Architecture.validation_network import FCN, CNN


class Agent:
    """Driver class for diversity network
    Attributes:
        Options (dict): configuration for the nn
    Methods:
        train_and_test: convienience function for preparing training and testing the model
    """

    def __init__(self, Options):
        """Sets up a new agent
        Args:
            Options (options): configuration options
        Returns:
            None
        """

        self.options = Options
        self.device = Options.device
        if Options.nn_type == "FCN":
            self.model = FCN(Options)
        elif Options.nn_type == "CNN":
            self.model = CNN(Options)
        self.dataset = Options.dataset
        self.train_loader, self.test_loader = self.dataset.process()

    def train_and_test(
        self,
    ) -> Tuple[Union[bool, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        """Trains and tests the model.

        Returns:
            Tuple[Union[bool, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]: Curve(if requested) and final test loss and accuracy
        """

        def _train() -> Union[bool, np.ndarray]:
            """Trains the model
            Args:
                None
            Returns:
                Union[bool, np.ndarray]: Traning curve
            """
            curve = False

            if self.options.save_curve is True:
                curve = np.empty((self.options.n_epochs, 5))

            for epoch_n in range(self.options.n_epochs):
                train_loss, train_acc = self.model.do_train(batches=self.train_loader)
                test_loss, test_acc = self.model.do_eval(batches=self.test_loader)
                if self.options.save_curve is True:
                    curve[epoch_n] = epoch_n, train_loss, train_acc, test_loss, test_acc

                if self.options.verbose:
                    print(
                        "Epoch: %03d, Train Loss: %0.4f, Train Train Acc: %0.4f, "
                        "Test Loss: %0.4f, Test Acc: %0.4f"
                        % (epoch_n, train_loss, train_acc, test_loss, test_acc)
                    )

            return curve

        def _test() -> Tuple[torch.Tensor, torch.Tensor]:
            """Evaluates the model on testing batches

            Returns:
                Tuple[torch.Tensor, torch.Tensor]: test accuracy and loss
            """
            (test_loss, test_acc) = self.model.do_eval(batches=self.test_loader)
            return test_loss, test_acc

        curve = _train()
        result = _test()

        if self.options.save_model and self.options.run_counter == 0:
            torch.save(
                self.model.state_dict(),
                self.options.output_folder
                + "/Saved_models/"
                + "{}_{}_{}_{}_saved_model_start_seed_{}_lr_{}_epochs_{}.pth".format(
                    self.options.dataset_name,
                    self.options.activation_name,
                    self.options.n_activations,
                    self.options.device,
                    self.options.seed,
                    self.options.inner_lr,
                    self.options.n_epochs,
                ),
            )
        return curve, result
