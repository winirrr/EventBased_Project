import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
from sklearn.model_selection import train_test_split



def plot_loss_curves(results):
    train_loss = results["train_loss"]
    train_acc = results["train_acc"]
    test_loss = results["test_loss"]
    test_acc = results["test_acc"]


    epochs = range(len(results["train_loss"]))

    # plt.figure(figsize=(10,5))
    # plt.plot(epochs, train_loss, label="train_loss")
    # plt.plot(epochs, test_loss, label="test_loss")
    # plt.title("Loss")
    # plt.xlabel("Epochs")
    # plt.legend()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
    ax[0].plot(epochs, train_loss, label="train_loss")
    ax[0].plot(epochs, test_loss, label="test_loss")
    ax[0].set_title("loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label="train_acc")
    ax[1].plot(epochs, test_acc, label="test_acc")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def save_model_std(model, model_name=None):
    MODEL_PATH = Path("models_save")
    MODEL_PATH.mkdir(exist_ok=True, parents=True)
    
    if model_name is None:
        model_name = model.__class__.__name__
    
    MODEL_NAME = f"{model_name}.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def count_files_in_directory(directory_path):
    # Dictionary to store the count of files for each label
    label_counts = {}

    # Iterate through each label folder in the directory
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            # Count the number of files in the label folder
            file_count = len([name for name in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, name))])
            label_counts[label] = file_count

    return label_counts


def create_file_label_dataframe(directory_path):
    file_paths = []
    labels = []

    #Iterate through each label folder in the directory
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
                    labels.append(label)

    # Create DataFrame
    df = pd.DataFrame(data={
        'file_name':file_paths,
        'label':labels
    })

    return df


def create_dfs(directory_path, majority_class_name, minority_class_name, num_minority_class):
    df = create_file_label_dataframe(directory_path=directory_path)
    sample_df = df.query(f'label == "{majority_class_name}" or label == "{minority_class_name}"')
    train_df, val_df = train_test_split(
        sample_df, stratify=sample_df.label, train_size=0.8, random_state=42
    )
    keep_images = set(
        train_df.query(f'label == "{minority_class_name}"').image.head(num_minority_class).tolist()
    )
    imbalanced_train_df = train_df[
        (train_df.image.isin(keep_images)) | (train_df.label == f"{majority_class_name}")
    ]
    return imbalanced_train_df, val_df



def plot_class_distribution(targets, title):
    class_counts = np.bincount(targets)
    colors = ['orange', 'blue']  # Specify colors for each class
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(class_counts)), class_counts, tick_label=[f'Class {i}' for i in range(len(class_counts))], color=colors)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_dataset_distribution(dataset, title):
    targets = [sample[1] for sample in dataset]
    plot_class_distribution(targets, title)


