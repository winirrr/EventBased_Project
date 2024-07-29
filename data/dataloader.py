import torch
import os
import numpy as np

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from data.custom_dataset import *
from utils.helper_function import *

NUM_WORKERS = 4  # Set the number of workers for DataLoader



def create_dataloader(
        data_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    # Create the dataset
    dataset = CustomDataset(root_dir=data_dir,transform=transform)
   
    # Define the split ratio
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate the lengths for training and testing
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for the training and test sets
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    # Check the dataset
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class to index mapping: {dataset.class_to_idx}")

    # Check sizes of the splits
    print(f"Total number of images: {dataset_size}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # Iterate over the training DataLoader
    print("Training DataLoader:")
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break

    # Iterate over the test DataLoader
    print("Test DataLoader:")
    for images, labels in test_dataloader:
        print(images.shape, labels.shape)
        break

    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    return train_dataloader, test_dataloader, dataset.classes


def create_dataloader_v2(
        data_dir: str,
        train_transform: transforms.Compose,
        test_transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    # Create the dataset
    dataset = CustomDataset(root_dir=data_dir)
   
    # Define the split ratio
    train_ratio = 0.8
    test_ratio = 0.2

    # Calculate the lengths for training and testing
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    test_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    # Create DataLoaders for the training and test sets
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    # Check the dataset
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class to index mapping: {dataset.class_to_idx}")

    # Check sizes of the splits
    print(f"Total number of images: {dataset_size}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # Iterate over the training DataLoader
    print("Training DataLoader:")
    for images, labels in train_dataloader:
        print(images.shape, labels.shape)
        break

    # Iterate over the test DataLoader
    print("Test DataLoader:")
    for images, labels in test_dataloader:
        print(images.shape, labels.shape)
        break

    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of test batches: {len(test_dataloader)}")
    return train_dataloader, test_dataloader, dataset.classes
