"""
Module to calculate the hessian of the classification model using pyhessian library.
https://github.com/amirgholami/PyHessian
"""
from pyhessian import hessian
import numpy as np
import torch
import argparse
import sys
import os

from density_plot import get_esd_plot

sys.path.extend(
    [f"../{name}" for name in os.listdir("../") if os.path.isdir("../" + name)]
)
from Source.Architecture.validation_network import FCN
from Source.DataLoading.DataProcessor import MNIST1D, MNIST, CIFAR
from Source.Helper.Dir import makepath

Description = "MetaLearningDiversity"
parser = argparse.ArgumentParser(description="model collector")

parser.add_argument(
    "--model_path",
    default="../Output/",
    type=str,
    help=" path to model for collecting hessian",
)
parser.add_argument(
    "--act_path",
    default="./Output/MNIST1D_sin_2_cuda:0_activations_start_seed_101_inner_lr_0.01_outer_lr_0.4_steps_5000_run_0.txt",
    type=str,
    help=" path to activation array for interpolation",
)
args = parser.parse_args()
model_path = args.model_path
act_path = args.act_path
base_path = "/".join(model_path.split("/")[:-2]) + "/"
file_name = model_path.split("/")[-1]
dataset_name = file_name.split("_")[0]
activation_name = file_name.split("_")[1]
n_activations = int(file_name.split("_")[2])
device = file_name.split("_")[3]
seed = int(file_name.split("_")[-5])
lr = float(file_name.split("_")[-3])
n_epochs = int(file_name.split("_")[-1].split(".")[0])

makepath(base_path + "Hessian/")


class options:
    """Hyperparameters and other configuration details for the neural network"""

    def __init__(self):
        self.dataset_name = dataset_name
        self.input_data = "./Data/"
        self.run_label = "Hessian"
        self.run_counter = 0
        self.training_split = 0.8
        self.inner_batch_size = 200
        self.inner_input_size = 40
        self.inner_hidden_size = 200
        self.inner_output_size = 10
        self.inner_lr = lr
        self.inner_mu = 0.8
        self.n_epochs = n_epochs
        self.n_activations = n_activations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.kwargs = {
            "batch_size": self.inner_batch_size,
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        self.seed = seed
        self.activation_name = activation_name
        self.act_path = act_path
        self.dataset = None
        self.validation_mode = False

    def __repr__(self):
        return f"{self.run_label} \nDataset: {self.dataset_name} \nactivation function: "\
            f"{self.n_activations}_{self.activation_name} \nnumber of epochs planned: {self.n_epochs}"\
                f"\ndevice: {self.device} \n"


def process_options(Options):

    # Initialize dataset objects
    if Options.dataset_name == "MNIST1D":
        Options.dataset = MNIST1D(Options)
    elif Options.dataset_name == "MNIST":
        Options.dataset = MNIST(Options)
    elif Options.dataset_name == "CIFAR":
        Options.dataset = CIFAR(Options)


Options = options()
process_options(Options)
print(Options)


model = FCN(Options)
dataset = Options.dataset
_, test_loader = dataset.process()

model_state = torch.load(model_path, map_location=Options.device)
model.load_state_dict(model_state)
criterion = torch.nn.CrossEntropyLoss()

for inputs, targets in test_loader:
    break
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
density_eigen, density_weight = hessian_comp.density()
# saving hessian data
np.savetxt(
    base_path
    + "Hessian/{}_{}_{}_lr_{}_epochs_{}_eigenvalues.txt".format(
        Options.dataset_name,
        Options.activation_name,
        str(Options.n_activations),
        Options.inner_lr,
        str(Options.n_epochs),
    ),
    density_eigen,
)
np.savetxt(
    base_path
    + "Hessian/{}_{}_{}_lr_{}_epochs_{}_weights.txt".format(
        Options.dataset_name,
        Options.activation_name,
        str(Options.n_activations),
        Options.inner_lr,
        str(Options.n_epochs),
    ),
    density_weight,
)
esd_plt = get_esd_plot(
    density_eigen,
    density_weight,
    base_path
    + "Hessian/{}_{}_{}_lr_{}_epochs_{}_density".format(
        Options.dataset_name,
        Options.activation_name,
        str(Options.n_activations),
        Options.inner_lr,
        str(Options.n_epochs),
    ),
)

torch.cuda.empty_cache()
