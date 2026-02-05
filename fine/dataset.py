#dataset.py
import os
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from sampling import iid_sampling, non_iid_dirichlet_sampling
import torch
from torch.utils.data import Subset
# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
def build_transforms(dataset_name: str, split: str):
    """
    Single transform per split.
    Training may include augmentation.
    Test is always deterministic.
    """
    dataset_name = dataset_name.lower()

    # CIFAR
    if dataset_name in ["cifar10", "cifar100"]:
        if split == "train":
            return transforms.Compose([
                transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    # OCT datasets
    if dataset_name in ["kermany", "olives", "srinivasan"]:
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])

    # Clothing1M
    if dataset_name == "clothing1m":
        if split == "train":
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


# ------------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------------
def get_dataset(dataset_name, num_users, iid, non_iid_prob_class, alpha_dirichlet, seed):
    """
    Returns:
      dataset_train
      dataset_test
      dict_users
      num_classes
    """

    name = dataset_name.lower()

    # ---------------- CIFAR ----------------
    if name == "cifar10":
        data_path = "../data/cifar10"
        num_classes = 10

        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=build_transforms(name, "train"))
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=build_transforms(name, "test"))
        y_train = np.array(dataset_train.targets)

        # y_train = np.array(dataset_train.targets)
        # y_test = np.array(dataset_test.targets)

        # # Keep only class 0 and 1
        # train_idx = np.where((y_train == 0) | (y_train == 1) | (y_train == 2))[0]
        # test_idx = np.where((y_test == 0) | (y_test == 1) | (y_test == 2))[0]

        # dataset_train = Subset(dataset_train, train_idx)
        # dataset_test = Subset(dataset_test, test_idx)

        # dataset_train.targets = y_train[train_idx]
        # dataset_test.targets = y_test[test_idx]
        # num_classes = 3
        

    elif name == "cifar100":
        data_path = "../data/cifar100"
        num_classes = 100

        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=build_transforms(name, "train"))
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=build_transforms(name, "test"))
        y_train = np.array(dataset_train.targets)

    # ---------------- Clothing1M ----------------
    elif name == "clothing1m":
        data_path = os.path.abspath("..") + "/data/clothing1M/"
        num_classes = 14

        dataset_train = Clothing(data_path, build_transforms(name, "train"), "train")
        dataset_test = Clothing(data_path, build_transforms(name, "test"), "test")
        y_train = np.array(dataset_train.targets)

    # ---------------- OCT ----------------
    elif name in ["kermany", "olives", "srinivasan"]:
        data_path = f"/data/OCT/classification/{dataset_name}"

        dataset_train = ImageFolder(root=data_path + "/train", transform=build_transforms(name, "train"))
        dataset_test = ImageFolder(root=data_path + "/test", transform=build_transforms(name, "test"))
        num_classes = len(dataset_train.classes)
        y_train = np.array(dataset_train.targets)

    else:
        raise ValueError(f"Unrecognized dataset: {dataset_name}")

    # ---------------- client partition ----------------
    n_train = len(dataset_train)
    if iid:
        dict_users = iid_sampling(n_train, num_users, seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, num_classes, non_iid_prob_class, num_users,seed, alpha_dirichlet)

    return dataset_train, dataset_test, dict_users, num_classes


class NoisyLabelDataset(torch.utils.data.Dataset):
    """
    Returns:
      x, y_noisy, y_clean, idx_global
    """
    def __init__(self, base_dataset, y_noisy, y_clean):
        self.base = base_dataset
        self.y_noisy = np.asarray(y_noisy).astype(int)
        self.y_clean = np.asarray(y_clean).astype(int)
        assert len(self.y_noisy) == len(self.base)
        assert len(self.y_clean) == len(self.base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]  # ignore base label
        return x, int(self.y_noisy[idx]), int(self.y_clean[idx]), int(idx)
