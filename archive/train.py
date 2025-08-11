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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import *
from dotenv import load_dotenv

load_dotenv()

labeling = f'andrew smoking labels'

fs = 50
window_size_seconds = 60
window_stride_seconds = 60
window_size = fs * window_size_seconds
window_stride = fs * window_stride_seconds

experiments = os.listdir('experiments')

if len(experiments) == 0:
    next_experiment = 1
else:
    next_experiment = max([int(exp.split('_')[0]) for exp in experiments if os.path.isdir(os.path.join('experiments', exp))]) + 1

hyperparameters = {
    'window_size': 3000,  # Size of the input window
    'num_features': 3,  # Number of features in the input data
    'batch_size': 512,  # Batch size for training and evaluation
    'learning_rate': 3e-4,  # Learning rate for the optimizer
    'weight_decay': 1e-4,  # Weight decay for regularization
    'num_epochs': 5000,  # Maximum number of training epochs
    'patience': 40,  # Early stopping patience,
    'experiment_name': f"{next_experiment}"
}

experiment_name = hyperparameters['experiment_name']
os.makedirs(f'experiments/{experiment_name}', exist_ok=True)

projects = get_projects_from_participant_codes(['P01','P02','P03','P04','P05','P06','P07','P08'])

print(projects)

X_train = []
y_train = []
X_test = []
y_test = []

for project_name in projects:
    print(f"Processing project: {project_name}")
    data = get_verified_and_not_deleted_sessions(project_name, labeling)
    project_path = get_project_path(project_name)
    sessions = data['sessions']

    if len(sessions) == 0:
        print(f"No verified sessions found for project {project_name} with labeling {labeling}.")
        continue

    train_sessions, test_sessions = train_test_split(data['sessions'], test_size=0.2, random_state=42)

    print(f"Train sessions size: {len(train_sessions)}")
    print(f"Test sessions size: {len(test_sessions)}")

    X,y = make_windowed_dataset_from_sessions(train_sessions, window_size, window_stride, project_path)
    X_train.append(X)
    y_train.append(y)

    X,y = make_windowed_dataset_from_sessions(test_sessions, window_size, window_stride, project_path)
    X_test.append(X)
    y_test.append(y)

X_train = torch.cat(X_train)
y_train = torch.cat(y_train)
X_test = torch.cat(X_test)
y_test = torch.cat(y_test)
torch.save((X_train,y_train),f'experiments/{experiment_name}/train.pt')
torch.save((X_test,y_test),f'experiments/{experiment_name}/test.pt')

print(f"Train dataset saved with shape: {X_train.shape}, {y_train.shape}")
print("Label distribution in train set:", y_train.long().bincount())
print(f"Test dataset saved with shape: {X_test.shape}, {y_test.shape}")
print("Label distribution in test set:", y_test.long().bincount())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SmokingCNN(window_size=hyperparameters['window_size'], num_features=hyperparameters['num_features'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
trainloader = DataLoader(TensorDataset(*torch.load(f'experiments/{experiment_name}/train.pt')), batch_size=hyperparameters['batch_size'], shuffle=True)
testloader = DataLoader(TensorDataset(*torch.load(f'experiments/{experiment_name}/test.pt')), batch_size=hyperparameters['batch_size'], shuffle=False)

trainlossi = []
testlossi = []
trainf1i = []
testf1i = []

n_epochs_without_improvement = 0
min_test_loss = float('inf')
max_test_f1 = 0.0
max_target_f1 = 0.0

model.to(device)

for epoch in range(5000):
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

    target_preds = []
    target_labels = []
    target_losses = []

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
                save_path=f'experiments/{experiment_name}/loss.jpg',
            )

        ## Early Stopping Logic on F1 Score
        if testf1i[-1] > max_test_f1:
            max_test_f1 = testf1i[-1]
            torch.save(model.state_dict(), f'experiments/{experiment_name}/best_model_base_f1.pt')
            print(f"Epoch {epoch}: New best model saved with test F1 score {max_test_f1:.4f}")
            n_epochs_without_improvement = 0
        else:
            print(f"Epoch {epoch}: No improvement in test F1 score ({testf1i[-1]:.4f} vs {max_test_f1:.4f})")
            n_epochs_without_improvement += 1
        if n_epochs_without_improvement >= hyperparameters['patience']:
            print(f"Early stopping triggered after {n_epochs_without_improvement} epochs without improvement.")
            n_epochs_without_improvement = 0
            break
            
        # Print epoch time and n_epochs_without_improvement
        epoch_time = time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds. "
            f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train F1: {trainf1i[-1]:.4f}, Test F1: {testf1i[-1]:.4f}. "
            f"Epochs without improvement: {n_epochs_without_improvement}")
        
torch.save(model.state_dict(),f'experiments/{experiment_name}/last.pt')

metrics = {
    'train_loss': trainlossi,
    'test_loss': testlossi,
    'train_f1': trainf1i,
    'test_f1': testf1i,
}

torch.save(metrics, f'experiments/{experiment_name}/metrics.pt')
