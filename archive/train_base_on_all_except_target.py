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

import argparse

parser = argparse.ArgumentParser(description='Train base model on all participants except target participant')
parser.add_argument('--target_participant', type=str, required=True, help='Target participant code (e.g., P01)')
args = parser.parse_args()

target_participant_code = args.target_participant
print(f"Target participant: {target_participant_code}")

def get_all_participants():
    """
    Fetch all participants from the database.
    Returns a list of dictionaries with participant details.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT participant_code FROM participants")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

participant_codes = get_all_participants()
participant_codes = [p for p in participant_codes if p != target_participant_code]
print(participant_codes)

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

projects = get_projects_from_participant_codes(participant_codes)

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
