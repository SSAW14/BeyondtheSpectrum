import warnings
from torchvision.datasets import *
from .base import *
from .image_dataset import BinaryImageDataset

datasets = {
    'image': BinaryImageDataset,
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)


