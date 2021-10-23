# -*- coding: utf-8 -*-
"""
Driver function for running the neural network
"""
import Trainer
import ValidationTrainer
import torch
import time
import numpy as np
import random
from Helper.Dir import makepath
from Helper.Parser import parse
from DataLoading.DataProcessor import MNIST1D, MNIST, CIFAR

Description = "MetaLearningDiversity_single_hiddenLayer_validation_tests"
args = parse(Description)  # parse cmd arguments to driver

#  Setting up the device
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
use_cuda = args.use_gpu and torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu_no) if use_cuda else "cpu")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
np.set_printoptions(precision=4, suppress=True)

kwargs = {"batch_size": args.inner_batch_size}
if use_cuda:
    kwargs.update(
        {"num_workers": 0, "pin_memory": False, "shuffle": True},
    )


class options:
    """Hyperparameters and other configuration details for the neural network"""

    def __init__(self, args):
        self.dataset_name = args.dataset
        self.input_data = args.data_path
        self.run_label = Description
        self.run_counter = 0
        self.training_split = 0.8
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_input_size = 1
        self.inner_input_size = 40
        self.outer_hidden_size = args.outer_hidden_size
        self.inner_hidden_size = args.inner_hidden_size
        self.outer_output_size = 1
        self.inner_output_size = 10
        self.outer_lr = args.outer_lr  # lr for outer loop with RMSprop
        self.inner_lr = args.inner_lr  # lr for classifier with SGD
        self.outer_mu = args.outer_mu
        self.inner_mu = args.inner_mu
        self.runs = args.runs
        self.n_epochs = args.epochs
        self.steps = args.steps
        self.n_activations = args.n_activations
        self.verbose = args.verbose
        self.device = device
        self.kwargs = kwargs
        self.seed = args.seed
        self.activation_name = args.activation
        self.act_path = args.act_path
        self.nn_type = args.nn_type
        self.validation_mode = args.validation_mode
        self.output_folder = (
            args.output_folder
            + "/"
            + Description
            + "/"
            + str(self.n_activations)
            + "_"
            + self.activation_name
            + "/"
        )
        self.save_curve = args.save_curve
        self.save_model = args.save_model
        self.dataset = None

    def __repr__(self):
        return f"{self.run_label} \nDataset: {self.dataset_name} \nactivation function: {self.activation_name} \n"\
            f"number of epochs planned: {self.n_epochs}\ndevice: {self.device} \n"


def process_options(Options):
    makepath(Options.input_data)

    makepath(Options.output_folder)

    if Options.save_curve is True:
        makepath(Options.output_folder + "Training_curves/")
    if Options.save_model is True:
        makepath(Options.output_folder + "Saved_models/")

    # Initialize dataset objects
    if Options.dataset_name == "MNIST1D":
        Options.dataset = MNIST1D(Options)
    elif Options.dataset_name == "MNIST":
        Options.dataset = MNIST(Options)
    elif Options.dataset_name == "CIFAR":
        Options.dataset = CIFAR(Options)

    return Options


if __name__ == "__main__":

    t0 = time.time()
    Options = options(args)
    if Options.act_path is None:
        process_options(Options)
        print(Options)
        results = np.empty((args.runs, 1))
        for i in range(args.runs):
            torch.manual_seed(Options.seed)
            torch.cuda.manual_seed_all(Options.seed)
            np.random.seed(Options.seed)
            random.seed(Options.seed)
            Options.seed += 1
            Options.run_counter = i
            agent = Trainer.Agent(Options)
            output = agent.outer_optimization()
            test_accuracy = output.inner_test_acc
            outer_losses = output.train_losses.cpu().numpy()
            activations = output.inner_afuncs
            grad_norms = output.grad_norms
            if Options.save_curve:
                np.savetxt(
                    Options.output_folder
                    + "Training_curves/"
                    + "{}_{}_{}_{}_outer_losses_start_seed_{}_inner_lr_{}_outer_lr_{}_steps_{}_run_{}.txt".format(
                        Options.dataset_name,
                        Options.activation_name,
                        str(Options.n_activations),
                        Options.device,
                        Options.seed,
                        str(Options.inner_lr),
                        str(Options.outer_lr),
                        str(Options.steps),
                        i,
                    ),
                    outer_losses,
                )
                np.savetxt(
                    Options.output_folder
                    + "Training_curves/"
                    + "{}_{}_{}_{}_test_accuracy_start_seed_{}_inner_lr_{}_outer_lr_{}_steps_{}_run_{}.txt".format(
                        Options.dataset_name,
                        Options.activation_name,
                        str(Options.n_activations),
                        Options.device,
                        Options.seed,
                        str(Options.inner_lr),
                        str(Options.outer_lr),
                        str(Options.steps),
                        i,
                    ),
                    test_accuracy,
                )
                np.savetxt(
                    Options.output_folder
                    + "{}_{}_{}_{}_grad_norms_start_seed_{}_inner_lr_{}_outer_lr_{}_steps_{}_run_{}.txt".format(
                        Options.dataset_name,
                        Options.activation_name,
                        str(Options.n_activations),
                        Options.device,
                        Options.seed,
                        str(Options.inner_lr),
                        str(Options.outer_lr),
                        str(Options.steps),
                        i,
                    ),
                    grad_norms,
                )

            np.savetxt(
                Options.output_folder
                + "{}_{}_{}_{}_activations_start_seed_{}_inner_lr_{}_outer_lr_{}_steps_{}_run_{}.txt".format(
                    Options.dataset_name,
                    Options.activation_name,
                    str(Options.n_activations),
                    Options.device,
                    Options.seed,
                    str(Options.inner_lr),
                    str(Options.outer_lr),
                    str(Options.steps),
                    i,
                ),
                activations,
            )
    if Options.act_path is not None:
        metadata = Options.act_path.split("/")[-1].split("_")
        # Options.activation_name = metadata[1]
        Options.n_activations = int(metadata[2])
        Options.seed = int(metadata[7])
        Options.inner_lr = float(metadata[10])
        Options.outer_lr = float(metadata[13])
        Options.steps = int(metadata[15])
        results = np.empty((args.runs, 2))
        validation = f"n{Options.validation_mode}" if Options.validation_mode else "n12"
        Options.output_folder = (
            args.output_folder
            + "/"
            + Description
            + "/"
            + Options.nn_type
            + "/"
            + f"{Options.n_activations}_{Options.activation_name}_validation_{validation}/"
        )
        process_options(Options)

        print("Validating activation function")
        print(
            f"{Options.run_label} \nDataset: {Options.dataset_name} \nactivation function: "\
                f"{Options.activation_name} \nnumber of epochs planned: {Options.n_epochs}\ndevice: {Options.device} \n"
        )
        for i in range(args.runs):
            print(f"Run#{i+1}")
            torch.manual_seed(Options.seed)
            torch.cuda.manual_seed_all(Options.seed)
            np.random.seed(Options.seed)
            random.seed(Options.seed)
            Options.seed += 1
            Options.run_counter = i
            agent = ValidationTrainer.Agent(Options)
            output = agent.train_and_test()

            training_curve = output[0]  # per epoch list of accuracy and losses

            results[i] = [
                output[1][0].item(),
                output[1][1].item(),
            ]  # test loss, test accuracy

            if Options.save_curve is True:
                np.savetxt(
                    Options.output_folder
                    + "Training_curves/"
                    + "{}_{}_{}_{}_train_loss_start_seed_{}_lr_{}_steps_{}_run_{}.txt".format(
                        Options.dataset_name,
                        Options.activation_name,
                        Options.n_activations,
                        Options.device,
                        Options.seed,
                        Options.inner_lr,
                        Options.steps,
                        i,
                    ),
                    training_curve,
                )

        np.savetxt(
            Options.output_folder
            + "{}_{}_{}_{}_test_accuracy_start_seed_{}_lr_{}_steps_{}.txt".format(
                Options.dataset_name,
                Options.activation_name,
                Options.n_activations,
                Options.device,
                Options.seed,
                Options.inner_lr,
                Options.steps,
            ),
            results,
        )
        torch.cuda.empty_cache()
    print("total runtime :", time.time() - t0)
