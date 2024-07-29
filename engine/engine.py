"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
 
    train_loss = 0
    train_acc = 0
    all_preds = []
    all_labels = []
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Store predictions and labels
        all_labels.append(y.cpu().numpy())
        all_preds.append(y_pred_class.cpu().numpy())

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss/ len(dataloader)
    train_acc = train_acc / len(dataloader)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return train_loss, train_acc, all_labels, all_preds

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.eval()

    test_loss = 0
    test_acc = 0
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate accuracy
            y_pred_class = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += (y_pred_class==y).sum().item() / len(test_pred_logits)

            # Store predictions and labels
            all_labels.append(y.cpu().numpy())
            all_preds.append(y_pred_class.cpu().numpy())

    # Adjust metrics to get average loss per batch
    test_loss = test_loss/len(dataloader)
    test_acc = test_acc / len(dataloader)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return test_loss, test_acc, all_labels, all_preds


class EarlyStopping():
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train(model: torch.nn.Module,
         train_dataloader: torch.utils.data.DataLoader,
         test_dataloader: torch.utils.data.DataLoader,
         optimizer: torch.optim.Optimizer,
         loss_fn: torch.nn.Module,
         epochs: int,
         device: torch.device,
         early_stopping=None,
         scheduler=None):
    """
    Trains and tests PyTorch model.
    """
    # Create empty results dictionary
    results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "train_precision" : [],
            "test_precision" : [],
            "train_recall" : [],
            "test_recall" : [],
            "train_confusion_matrix" : [],
            "test_confusion_matrix" : []
    }

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_labels, train_preds= train_step(model=model,
                                                                     dataloader=train_dataloader,
                                                                     loss_fn=loss_fn,
                                                                     optimizer=optimizer,
                                                                     device=device)
        
        test_loss, test_acc, test_labels, test_preds = test_step(model=model,
                                                                 dataloader=test_dataloader,
                                                                 loss_fn=loss_fn,
                                                                 device=device)
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epochs: {epoch} | train_loss : {train_loss:.4f} | train_acc: {train_acc:.2%} | test_loss : {test_loss:.4f}| test_acc: {test_acc:.2%}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Compute precision and recall
        train_precision = precision_score(train_labels, train_preds, average="weighted", zero_division=0)
        test_precision = precision_score(test_labels, test_preds, average="weighted", zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average="weighted", zero_division=0)
        test_recall = recall_score(test_labels, test_preds, average="weighted", zero_division=0)

        results["train_precision"].append(train_precision)
        results["test_precision"].append(test_precision)
        results["train_recall"].append(train_recall)
        results["test_recall"].append(test_recall)

        # Compute confusion matrix
        train_cm = confusion_matrix(train_labels, train_preds)
        test_cm = confusion_matrix(test_labels, test_preds)
        
        results["train_confusion_matrix"].append(train_cm)
        results["test_confusion_matrix"].append(test_cm)

        # Check for Early stopping
        if early_stopping is not None:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break

    return results
