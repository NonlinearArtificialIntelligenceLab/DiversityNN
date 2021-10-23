import argparse
from argparse import Namespace


def parse(description: str) -> Namespace:
    """Parses commandline arguments given to the driver

    Args:
        description (str): description of network task
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--use_gpu", dest="use_gpu", action="store_true", help="To GPU or not to GPU"
    )
    parser.add_argument("--gpu_no", default=0, type=int, help="Which GPU to use ?")
    parser.add_argument("--data_path", default="./Data/", type=str, help="Path to data")
    parser.add_argument(
        "--output_folder", default="./Output", type=str, help="Path to output folder"
    )
    parser.add_argument(
        "--inner_batch_size", default=200, type=int, help="Batch size for classifier"
    )
    parser.add_argument(
        "--outer_batch_size",
        default=1,
        type=int,
        help="Batch size for activation function",
    )
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
    parser.add_argument(
        "--steps", default=300, type=int, help="Number of steps in outer loop"
    )
    parser.add_argument(
        "--inner_lr", default=1e-02, type=float, help="Learning rate for classifier"
    )
    parser.add_argument(
        "--outer_lr",
        default=0.4,
        type=float,
        help="Learning rate for activation function",
    )
    parser.add_argument(
        "--inner_mu", default=0.8, type=float, help="SGD momentum for classifier"
    )
    parser.add_argument(
        "--outer_mu",
        default=0.8,
        type=float,
        help="SGD momentum for activation function",
    )
    parser.add_argument(
        "--inner_hidden_size",
        default=200,
        type=int,
        help="Hidden Layer size for classifier",
    )
    parser.add_argument(
        "--outer_hidden_size",
        default=50,
        type=int,
        help="Hidden Layer size for activation function",
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--seed", default=100, type=int, help="fixed seed for reproducibility"
    )
    parser.add_argument(
        "--runs", default=1, type=int, help="No. of runs in configuration"
    )
    parser.add_argument(
        "--dataset",
        default="MNIST1D",
        type=str,
        help=" Select dataset from MNIST1D, MNIST, CIFAR",
    )
    parser.add_argument(
        "--nn_type",
        default="FCN",
        type=str,
        help=" Select network type from FCN",
    )
    parser.add_argument(
        "--validation_mode",
        default=False,
        type=int,
        help=" select validation mode from activation num or false for all",
    )
    parser.add_argument(
        "--activation",
        default="sin",
        type=str,
        help=" Select base activation function to be modified. Choose from elu, sin, tanh",
    )
    parser.add_argument(
        "--act_path",
        default=None,
        type=str,
        help=" Give path to activation function for validation",
    )
    parser.add_argument(
        "--n_activations",
        default=1,
        type=int,
        help="No. of activation functions to be used in classifier",
    )
    parser.add_argument(
        "--save_curve", action="store_true", help="saves model training curve"
    )
    parser.add_argument(
        "--save_model", action="store_true", help="saves model state dict"
    )
    args = parser.parse_args()

    return args
