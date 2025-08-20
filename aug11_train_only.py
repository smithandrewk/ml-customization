"""
train test split across sessions for multiple participants
"""
import os
import json
from tqdm import tqdm
from time import time

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *
from dotenv import load_dotenv

load_dotenv()

fs = 50
window_size_seconds = 60
window_stride_seconds = 60
window_size = fs * window_size_seconds
window_stride = fs * window_stride_seconds


hyperparameters = {
    'window_size': 3000,
    'num_features': 3,
    'batch_size': 512,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    'num_epochs': 5000,
    'patience': 40,
    'experiment_name': "2"
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create experiment directory if it doesn't exist
os.makedirs(f'experiments/{hyperparameters["experiment_name"]}', exist_ok=True)

model = SmokingCNN(window_size=hyperparameters['window_size'], num_features=hyperparameters['num_features'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
trainloader = DataLoader(TensorDataset(*torch.load(f'experiments/{hyperparameters["experiment_name"]}/train.pt')), batch_size=hyperparameters['batch_size'], shuffle=True)
testloader = DataLoader(TensorDataset(*torch.load(f'experiments/{hyperparameters["experiment_name"]}/test.pt')), batch_size=hyperparameters['batch_size'], shuffle=False)

# Get dataset sizes
train_dataset_size = len(trainloader.dataset)
test_dataset_size = len(testloader.dataset)

# Print dataset information
print(f"Dataset Information:")
print(f"Training samples: {train_dataset_size:,}")
print(f"Test samples: {test_dataset_size:,}")
print(f"Total samples: {train_dataset_size + test_dataset_size:,}")
print(f"Train/Test split: {train_dataset_size/(train_dataset_size + test_dataset_size):.1%}/{test_dataset_size/(train_dataset_size + test_dataset_size):.1%}")
print()

trainlossi = []
testlossi = []
trainf1i = []
testf1i = []

# Initialize tracking variables (separate from hyperparameters to avoid mutation)
n_epochs_without_improvement = 0
min_test_loss = float('inf')
max_test_f1 = 0.0

model.to(device)

for epoch in range(hyperparameters['num_epochs']):
    epoch_start_time = time()
    model.train()

    train_preds = []
    train_labels = []
    train_losses = []

    for Xi, yi in trainloader:
        Xi, yi = Xi.to(device), yi.to(device)
        optimizer.zero_grad()
        outputs = model(Xi).squeeze()
        loss = criterion(outputs, yi)
        loss.backward()
        optimizer.step()

        train_preds.append(outputs.sigmoid().round().cpu())
        train_labels.append(yi.cpu())
        train_losses.append(loss.item())

    train_preds = torch.cat(train_preds).detach()
    train_labels = torch.cat(train_labels)
    train_loss = np.mean(train_losses)
    trainlossi.append(train_loss)
    trainf1i.append(f1_score(train_labels, train_preds, average='macro'))

    model.eval()

    test_preds = []
    test_labels = []
    test_losses = []

    with torch.no_grad():
        for Xi, yi in testloader:
            Xi, yi = Xi.to(device), yi.to(device)
            outputs = model(Xi).squeeze()
            loss = criterion(outputs, yi)

            test_preds.append(outputs.sigmoid().round().cpu())
            test_labels.append(yi.cpu())
            test_losses.append(loss.item())

        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_loss = np.mean(test_losses)
        testlossi.append(test_loss)
        testf1i.append(f1_score(test_labels, test_preds, average='macro'))

        if epoch % 1 == 0:
            plot_training_progress(
                trainlossi=trainlossi,
                testlossi=testlossi,
                trainf1i=trainf1i,
                testf1i=testf1i,
                ma_window_size=5,
                save_path=f'experiments/{hyperparameters["experiment_name"]}/loss.jpg',
            )

        ## Early Stopping Logic on F1 Score
        if testf1i[-1] > max_test_f1:
            max_test_f1 = testf1i[-1]
            torch.save(model.state_dict(), f'experiments/{hyperparameters["experiment_name"]}/best_model_base_f1.pt')
            print(f"Epoch {epoch}: New best model saved with test F1 score {max_test_f1:.4f}")
            n_epochs_without_improvement = 0
        else:
            print(f"Epoch {epoch}: No improvement in test F1 score ({testf1i[-1]:.4f} vs {max_test_f1:.4f})")
            n_epochs_without_improvement += 1
        if n_epochs_without_improvement >= hyperparameters['patience']:
            print(f"Early stopping triggered after {n_epochs_without_improvement} epochs without improvement.")
            break
            
        # Print epoch time and n_epochs_without_improvement
        epoch_time = time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds. "
            f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train F1: {trainf1i[-1]:.4f}, Test F1: {testf1i[-1]:.4f}. "
            f"Epochs without improvement: {n_epochs_without_improvement}")
        
# Final model save with training summary
print(f"\nTraining completed!")
print(f"Best F1 Score achieved: {max_test_f1:.4f}")
print(f"Final model saved to: experiments/{hyperparameters['experiment_name']}/last.pt")

torch.save(model.state_dict(),f'experiments/{hyperparameters["experiment_name"]}/last.pt')

metrics = {
    'train_loss': trainlossi,
    'test_loss': testlossi,
    'train_f1': trainf1i,
    'test_f1': testf1i,
    'hyperparameters': hyperparameters,
    'best_f1': max_test_f1,
    'train_samples': train_dataset_size,
    'test_samples': test_dataset_size,
    'total_samples': train_dataset_size + test_dataset_size
}

torch.save(metrics, f'experiments/{hyperparameters["experiment_name"]}/metrics.pt')
