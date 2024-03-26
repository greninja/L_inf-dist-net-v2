import numpy as np
import random
import glob
import torch
import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST, ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2860]),
    'CIFAR10': np.array([0.4914, 0.4822, 0.4465]),
    'TinyImageNet': np.array([0.4802, 0.4481, 0.3975]),
}
std = {
    'MNIST': 0.3081,
    'FashionMNIST': 0.3520,
    'CIFAR10': 0.2009, #np.array([0.2023, 0.1994, 0.2010])
    'TinyImageNet': 0.2276,#[0.2302, 0.2265, 0.2262]
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'FashionMNIST': [transforms.RandomCrop(28, padding=1, padding_mode='edge')],
    'CIFAR10': [transforms.RandomCrop(32, padding=3, padding_mode='edge'), transforms.RandomHorizontalFlip()],
    # 'TinyImageNet': [transforms.RandomHorizontalFlip(), transforms.RandomCrop(56)],
    'TinyImageNet': [transforms.RandomCrop(64, padding=4, padding_mode='edge'), transforms.RandomHorizontalFlip()],
}
test_transforms = {
    'MNIST': [],
    'FashionMNIST': [],
    'CIFAR10': [],
    # 'TinyImageNet': [transforms.CenterCrop(56)],
    'TinyImageNet': [],
}
input_dim = {
    'MNIST': np.array([1, 28, 28]),
    'FashionMNIST': np.array([1, 28, 28]),
    # 'CIFAR10': np.array([3, 32, 32]),
    'CIFAR10': np.array([3, 48, 48]),
    # 'TinyImageNet': np.array([3, 56, 56]),
    'TinyImageNet': np.array([3, 64, 64]),
}
default_eps = {
    'MNIST': 0.3,
    'FashionMNIST': 0.1,
    'CIFAR10': 0.03137,
    'TinyImageNet': 1. / 255,
}


def get_statistics(dataset):
    return mean[dataset], std[dataset]

class PadAndShift(object):
    def __init__(self, transform_params: dict = {}):
        # add parameters of the location of the image
        self.pad_size = transform_params["pad_size"] # one sided pad_size; hence total padding = 2 * pad_size
        self.num_image_locations = transform_params["num_image_locations"]
        self.background = transform_params["background"] # options = ["black", "nature"]

        if self.background == "nature_5bg":
            # 5 images sampled from 20kBG images
            # no train/test split
            # self.random_bg_image_paths = glob.glob("/home/ss3shaik/projects/def-mlecuyer/ss3shaik/bg_20k/images/5_imgs/48_48_size/*")
            self.random_bg_image_paths = glob.glob("/data1/shadabs3/bg_20k/images/5_imgs/48_48_size/*")
        # elif self.background == "nature_20kbg":
        #     # All 20kBG images
        #     # with train/test split
        #     if split == "train":
        #         # only sample from bg-20k train dataset (48x48x3 images)
        #         self.random_bg_image_paths = glob.glob("/home/ss3shaik/projects/def-mlecuyer/ss3shaik/bg_20k/images/only_train/48_48_size/*")
        #     elif split == "test":
        #         # only sample from bg-20k test dataset (48x48x3 images)
        #         self.random_bg_image_paths = glob.glob("/home/ss3shaik/projects/def-mlecuyer/ss3shaik/bg_20k/images/only_test/48_48_size/*")

    def __call__(self, image):
        # assume the image is of PIL form, hence first convert it to a numpy array
        image = np.array(image)
        
        h, w, _ = image.shape
        new_h = h + 2 * self.pad_size
        new_w = w + 2 * self.pad_size

        # generate the padded image based on chosen background
        if self.background == "black" or self.background == None:
            padded_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        elif "nature" in self.background:
            random_bg_image_path = random.choice(self.random_bg_image_paths)
            bg_img = Image.open(random_bg_image_path)
            bg_img = bg_img.resize((new_w, new_h))
            padded_image = np.array(bg_img)
        # elif self.background == "replicate":
        #     # can't have a static background
        #     pass

        # if shifting, choose random locations or static location if not
        if self.pad_size > 0:
            if self.num_image_locations == "1":
                # center location; statically place the image in the center
                choices = [(self.pad_size, self.pad_size)]
                x_prime, y_prime = random.choice(choices)
            elif self.num_image_locations == "2":
                # top left or bottom right
                choices = [(0,0),
                           (new_h - h, new_w - w)]
                x_prime, y_prime = random.choice(choices)
            elif self.num_image_locations == "4":
                # top left, top right, bottom left, bottom right
                choices = [(0, 0),
                           (new_h - h, 0),
                           (0, new_w - w),
                           (new_h - h, new_w - w)]
                x_prime, y_prime = random.choice(choices)
            elif self.num_image_locations == "8":
                # top left, top right, bottom left, bottom right,
                # center top, center right, center left, center bottom
                choices = [
                           (0, 0),
                           (new_h - h, 0),
                           (0, new_w - w),
                           (new_h - h, new_w - w),
                           (0, self.pad_size),
                           (new_h - h, self.pad_size),
                           (self.pad_size, 0),
                           (self.pad_size, new_w - w),
                          ]
                x_prime, y_prime = random.choice(choices)
            elif self.num_image_locations == "edges":
                a = random.choice([0, 2*self.pad_size])
                b = random.choice(np.arange(0, 2*self.pad_size))
                flip = np.random.binomial(1,0.5)
                if flip == 0:
                    x_prime, y_prime = a, b
                else:
                    x_prime, y_prime = b, a
            elif self.num_image_locations == "random":
                # place image randomly anywhere within the padded image
                x_prime, y_prime = np.random.randint(low=0, high=2*self.pad_size, size=2)
            else:
                raise Exception("Choose between 1, 2, 4, 8, edges, random")
        else:
            x_prime, y_prime = 0, 0
        
        # put the image as per shifting choice
        if self.background in ["black", "nature_5bg", "nature_20kbg", None]:
            padded_image[x_prime:x_prime + h, y_prime:y_prime + w, :] = image
        # elif self.background in ["replicate"]:
        #     padding_values = (x_prime, # pad_left
        #                       2*self.pad_size - x_prime, # pad_right
        #                       y_prime, # pad_top
        #                       2*self.pad_size - y_prime) # pad_bottom
        #     image = torch.from_numpy(image)
        #     print(image.shape)
        #     padded_image = F.pad(image, padding_values, mode='replicate')

        return padded_image

class TinyImageNet(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None, **kwargs):
        path = 'train' if train else 'val'
        self.data = ImageFolder(os.path.join('tiny-imagenet-200', path), transform=transform)
        self.classes = self.data.classes
        self.transform = transform

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

def get_dataset(dataset, dataset_name, datadir, augmentation=True):
    transform_params = {
        "pad_size": 8,
        "num_image_locations": "edges", 
        "background": "nature_5bg"
    }
    train_transform = transforms.Compose([
        PadAndShift(transform_params),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean[dataset_name], [std[dataset_name]] * len(mean[dataset_name]))
    ])
    test_transform = transforms.Compose([
        PadAndShift(transform_params),
        transforms.ToTensor(),
    ])
    Dataset = globals()[dataset]
    train_dataset = Dataset(root=datadir, train=True, download=True, transform=train_transform)
    test_dataset = Dataset(root=datadir, train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset

def load_data(dataset, datadir, batch_size, parallel, augmentation=True, workers=4):
    train_dataset, test_dataset = get_dataset(dataset, dataset, datadir, augmentation=augmentation)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=torch.seed()) if parallel else None
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=not parallel,
                             num_workers=workers, sampler=train_sampler, pin_memory=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if parallel else None
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, sampler=test_sampler, pin_memory=True)
    return train_dataset, trainloader, test_dataset, testloader
