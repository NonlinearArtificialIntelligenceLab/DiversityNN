"""Module for generating and manipulating MNIST-1D dataset
The MNIST-1D dataset | 2020
Sam Greydanus https://github.com/greydanus/mnist1d
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import torch
import random
import pickle
import pathlib
from torch.utils.data.dataset import Dataset
import requests

# from PIL import Image
# import torchvision

# transformations of the templates which will make them harder to classify
def pad(x, padding):
    low, high = padding
    p = low + int(np.random.rand() * (high - low + 1))
    return np.concatenate([x, np.zeros((p))])


def shear(x, scale=10):
    coeff = scale * (np.random.rand() - 0.5)
    return x - coeff * np.linspace(-0.5, 0.5, len(x))


def translate(x, max_translation):
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])


def corr_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return gaussian_filter(noise, 2)


def iid_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return noise


def interpolate(x, N):
    scale = np.linspace(0, 1, len(x))
    new_scale = np.linspace(0, 1, N)
    new_x = interp1d(scale, x, axis=0, kind="linear")(new_scale)
    return new_x


def transform(x, y, args, eps=1e-8):
    new_x = pad(x + eps, args.padding)  # pad
    new_x = interpolate(new_x, args.template_len + args.padding[-1])  # dilate
    new_y = interpolate(y, args.template_len + args.padding[-1])
    new_x *= 1 + args.scale_coeff * (np.random.rand() - 0.5)  # scale
    new_x = translate(new_x, args.max_translation)  # translate

    # add noise
    mask = new_x != 0
    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, args.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)

    # shear and interpolate
    new_x = shear(new_x, args.shear_scale)
    new_x = interpolate(new_x, args.final_seq_length)  # subsample
    new_y = interpolate(new_y, args.final_seq_length)
    return new_x, new_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_pickle(thing, path):  # save something
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=3)


def from_pickle(path):  # load something
    thing = None
    with open(path, "rb") as handle:
        thing = pickle.load(handle)
    return thing


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


def get_dataset_args(as_dict=False):
    arg_dict = {
        "num_samples": 10000,  # modified for meta-learning
        "train_split": 0.8,
        "template_len": 12,
        "padding": [36, 60],
        "scale_coeff": 0.4,
        "max_translation": 48,
        "corr_noise_scale": 0.25,
        "iid_noise_scale": 2e-2,
        "shear_scale": 0.75,
        "final_seq_length": 40,
        "url": "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl",
    }
    return arg_dict if as_dict else ObjectView(arg_dict)


# basic 1D templates for the 10 digits
def get_templates():
    d0 = np.asarray([5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5])
    d1 = np.asarray([5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5])
    d2 = np.asarray([5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5])
    d3 = np.asarray([5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5])
    d4 = np.asarray([5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5])
    d5 = np.asarray([5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5])
    d6 = np.asarray([5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5])
    d7 = np.asarray([5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5])
    d8 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5])
    d9 = np.asarray([5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5])

    x = np.stack([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])
    x -= x.mean(1, keepdims=True)  # whiten
    x /= x.std(1, keepdims=True)
    x -= x[:, :1]  # signal starts and ends at 0

    templates = {
        "x": x / 6.0,
        "t": np.linspace(-5, 5, len(d0)) / 6.0,
        "y": np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    }
    return templates


# make a dataset
def make_dataset(
    args=None,
    template=None,
):
    templates = get_templates() if template is None else template
    args = get_dataset_args() if args is None else args
    xs, ys = [], []
    samples_per_class = args.num_samples // len(templates["y"])
    for label_ix in range(len(templates["y"])):
        for example_ix in range(samples_per_class):
            x = templates["x"][label_ix]
            t = templates["t"]
            y = templates["y"][label_ix]
            x, new_t = transform(x, t, args)  # new_t transformation is same each time
            xs.append(x)
            ys.append(y)

    batch_shuffle = np.random.permutation(len(ys))  # shuffle batch dimension
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]

    new_t = new_t / xs.std()
    xs = (xs - xs.mean()) / xs.std()  # center the dataset & set standard deviation to 1

    # train / test split
    split_ix = int(len(ys) * args.train_split)
    dataset = {
        "x": xs[:split_ix],
        "x_test": xs[split_ix:],
        "y": ys[:split_ix],
        "y_test": ys[split_ix:],
        "t": new_t,
        "templates": templates,
    }
    return dataset


class MNIST1DDataset(Dataset):
    """Pytorch custom dataset for loading 1d-mnist dataset"""

    def __init__(
        self,
        root_path,
        train,
        regenerate,
        download,
        transform=None,
        target_transform=None,
        **kwargs
    ):
        """
        Args:
            root_path (string): path to the dataset pickle file
            train (bool): selects between training and testing dataset
            regenerate (bool): Whether to regenerate the dataset
            download (bool): Downloads original 1D mnist dataset
            transform (Function): Applies transformation to PIL image
            target_transform (Function): Applies transformation to target values
        """
        self.args = get_dataset_args()
        self.path = root_path + "mnist1d_data.pkl"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        assert not (
            download and regenerate
        ), "You can either download the o.g. MNIST1D dataset or generate your own - but not both"

        if pathlib.Path(self.path).exists() and not regenerate:
            self.dataset = from_pickle(self.path)
        elif regenerate:
            print("regenerating 1D mnist dataset")
            self.dataset = make_dataset(self.args, **kwargs)
            if not pathlib.Path(self.path).exists:
                to_pickle(self.dataset, self.path)
                print("Saving to {}".format(self.path))
            self.dataset = from_pickle(self.path)
        elif download:
            print("Downloading original 1D mnist dataset")
            r = requests.get(self.args.url, allow_redirects=True)
            open(self.path, "wb").write(r.content)
            print("Saving to {}".format(self.path))
            self.dataset = from_pickle(self.path)
        if train:
            self.imgs = self.dataset["x"]
            self.targets = self.dataset["y"]
        else:
            self.imgs = self.dataset["x_test"]
            self.targets = self.dataset["y_test"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, target = self.imgs[idx].astype("float32"), int(self.targets[idx])
        # img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":

    def _process_IDMNIST():

        """Process MNIST for use by the network"""
        train_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path="../../data/",
                train=True,
                regenerate=False,
                download=True,
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            MNIST1DDataset(
                root_path="../../data/",
                train=False,
                regenerate=False,
                download=True,
            ),
        )
        return train_loader, test_loader

    a, b = _process_IDMNIST()
