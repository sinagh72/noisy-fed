from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from sampling import iid_sampling, non_iid_dirichlet_sampling
import torch.utils
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

def get_dataset(dataset_name, num_users, iid, non_iid_prob_class, alpha_dirichlet, seed):
    if dataset_name == 'cifar10':
        data_path = '../data/cifar10'
        num_classes = 10
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif dataset_name == 'cifar100':
        data_path = '../data/cifar100'
        num_classes = 100
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif dataset_name == 'clothing1m':
        data_path = os.path.abspath('..') + '/data/clothing1M/'
        num_classes = 14
        trans_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        trans_val = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 ])
        dataset_train = Clothing(data_path, trans_train, "train")
        dataset_test = Clothing(data_path, trans_val, "test")
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    elif dataset_name == "kermany":
        data_path = '/data/OCT/classification/Kermany'
        num_classes = 4
        trans_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        trans_test = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        dataset_train = ImageFolder(root=data_path+"/train", transform=trans_train)
        dataset_test = ImageFolder(root=data_path+"/test", transform=trans_test)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif dataset_name == "OLIVES":
        data_path = '/data/OCT/classification/OLIVES'
        num_classes = 2
        trans_train = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        trans_test = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                 ])
        dataset_train = ImageFolder(root=data_path+"/train", transform=trans_train)
        dataset_test = ImageFolder(root=data_path+"/test", transform=trans_test)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    else:
        exit('Error: unrecognized dataset')

    if iid:
        dict_users = iid_sampling(n_train, num_users, seed)
    else:
        dict_users = non_iid_dirichlet_sampling(y_train, num_classes, non_iid_prob_class, num_users, seed, alpha_dirichlet)
    return dataset_train, dataset_test, dict_users, num_classes


class Clothing(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.noisy_labels = {}
        self.clean_labels = {}
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        with open(self.root + 'noisy_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.noisy_labels[img_path] = int(entry[1])

        with open(self.root + 'clean_label_kv.txt', 'r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()
            img_path = self.root + entry[0]
            self.clean_labels[img_path] = int(entry[1])

        if self.mode == 'train':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'minitrain':
            with open(self.root + 'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            n = len(lines)
            np.random.seed(13)
            subset_idx = np.random.choice(n, int(n/10), replace=False)
            for i in subset_idx:
                l = lines[i]
                img_path = self.root + l
                self.data.append(img_path)
                target = self.noisy_labels[img_path]
                self.targets.append(target)
        elif self.mode == 'test':
            with open(self.root + 'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = self.root + l
                self.data.append(img_path)
                target = self.clean_labels[img_path]
                self.targets.append(target)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)


class NoisyLabelDataset(Dataset):
    """
    Wraps a torchvision dataset so __getitem__ returns:
      x, y_noisy, idx, y_clean
    """
    def __init__(self, base_dataset, clean_targets):
        self.base = base_dataset
        self.clean_targets = torch.as_tensor(clean_targets, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y_noisy = self.base[idx]  # base uses noisy targets already
        y_clean = self.clean_targets[idx].item()
        return x, int(y_noisy), idx, int(y_clean)