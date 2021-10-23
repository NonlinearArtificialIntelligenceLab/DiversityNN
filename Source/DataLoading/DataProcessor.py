from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torchvision
from .MNIST1D import MNIST1DDataset


class DataProcessor(ABC, nn.Module):
    """
    Data and manipulation tools for the NN
    """

    def __init__(self, Options):
        super(DataProcessor, self).__init__()
        self.name = Options.dataset_name
        self.options = Options
        self.device = self.options.device

    @abstractmethod
    def process(self):from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn as nn
import torchvision
from .MNIST1D import MNIST1DDataset


class DataProcessor(ABC, nn.Module):
    """
    Data and manipulation tools for the NN
    """

    def __init__(self, Options):
        super(DataProcessor, self).__init__()
        self.name = Options.dataset_name
        self.options = Options
        self.device = self.options.device

    @abstractmethod
    def process(self):
        """
        Process dataset for use with the trainer
        """
        pass


class MNIST1D(DataProcessor):
    """MNIST 1D dataset and manipulation tools"""

    def __init__(self, options):
        super().__init__(options)
        self.input_size = 40
        self.output_size = 10

    def process(self):
        """Process IDMNIST for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path=self.options.input_data,
                train=True,
                regenerate=False,
                download=True,
            ),
            **self.options.kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path=self.options.input_data,
                train=False,
                regenerate=False,
                download=True,
            ),
            **self.options.kwargs
        )
        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes MNIST1D data for FC use
        Args:
            data (torch.Tensor): input data
        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 40 * 1)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use
        Args:
            data (torch.Tensor):input data
        Returns:
            torch.Tensor: reshaped data
        """
        return data.view(-1, 1, data.shape[-1])


class MNIST(DataProcessor):
    """MNIST dataset and manipulation tools"""

    def __init__(self, options):
        super().__init__(options)
        self.input_size = 784
        self.output_size = 10

    def process(self):
        """Process MNIST for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.options.input_data,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.options.input_data,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )
        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes MNIST data for FC use

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 28 * 28)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use

        Args:
            data (torch.Tensor):input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data


class CIFAR(DataProcessor):
    """CIFAR dataset and manipulation tools"""

    def __init__(self, options):
        super().__init__(options)
        self.input_size = 3072
        self.output_size = 10

    def process(self):
        """Process CIFAR for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                self.options.input_data,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                self.options.input_data,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes CIFAR data for FC use

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 3 * 32 * 32)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use

        Args:
            data (torch.Tensor):input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data

        """
        Process dataset for use with the trainer
        """
        pass


class MNIST1D(DataProcessor):
    """MNIST 1D dataset and manipulation tools"""

    def __init__(self, options):
        super().__init__(options)
        self.input_size = 40
        self.output_size = 10

    def process(self):
        """Process IDMNIST for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path=self.options.input_data,
                train=True,
                regenerate=False,
                download=True,
            ),
            **self.options.kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path=self.options.input_data,
                train=False,
                regenerate=False,
                download=True,
            ),
            **self.options.kwargs
        )
        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes MNIST1D data for FC use
        Args:
            data (torch.Tensor): input data
        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 40 * 1)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use
        Args:
            data (torch.Tensor):input data
        Returns:
            torch.Tensor: reshaped data
        """
        return data.view(-1, 1, data.shape[-1])


class MNIST(DataProcessor):
    def process(self):
        """Process MNIST for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.options.input_data,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.options.input_data,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )
        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes MNIST data for FC use

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 28 * 28)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use

        Args:
            data (torch.Tensor):input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data


class CIFAR(DataProcessor):
    def process(self):
        """Process CIFAR for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                self.options.input_data,
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                self.options.input_data,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                    ]
                ),
            ),
            **self.options.kwargs
        )

        return train_loader, test_loader

    def fc_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """Reshapes CIFAR data for FC use

        Args:
            data (torch.Tensor): input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data.reshape(-1, 3 * 32 * 32)

    def conv_reshape(self, data: torch.Tensor) -> torch.Tensor:
        """reshapes data for CNN use

        Args:
            data (torch.Tensor):input data

        Returns:
            torch.Tensor: reshaped data
        """
        return data
