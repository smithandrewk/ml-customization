import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import argparse
from dotenv import load_dotenv
from sympy import hyper
import torch
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from time import time

load_dotenv()

parser = argparse.ArgumentParser(description='Train a smoking detection model')
parser.add_argument('--participant_code', type=str, default='all', help='Participant code to train on (default: all)')
# batch size
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation (default: 512)')
args = parser.parse_args()
participant_code = args.participant_code

next_experiment = get_next_experiment_number()

hyperparameters = {
    'window_size': 3000,  # Size of the input window
    'num_features': 3,  # Number of features in the input data
    'batch_size': args.batch_size,  # Batch size for training and evaluation
    'learning_rate': 3e-4,  # Learning rate for the optimizer
    'weight_decay': 1e-4,  # Weight decay for regularization
    'num_epochs': 5000,  # Maximum number of training epochs
    'patience': 40,  # Early stopping patience,
    'experiment_name': f"{next_experiment}_{participant_code}"
}

labeling = f'smoking'
fs = 50
window_size_seconds = 60
window_stride_seconds = 60
# if window_size_seconds == window_stride_seconds then we can split across windows
window_size = fs * window_size_seconds
window_stride = fs * window_stride_seconds

projects = []

for participant_code in ['kerry']:
    participant_id = get_participant_id(participant_code)
    participant_projects = get_participant_projects(participant_id)
    if len(participant_projects) == 0:
        print(f"No projects found for participant {participant_code}.")
        continue
    print(f"Participant {participant_code} has projects: {participant_projects}")
    projects.extend(participant_projects)

print(projects)

X = []
y = []

for project_name in projects:
    print(f"Processing project: {project_name}")
    data = get_verified_and_not_deleted_sessions(project_name, labeling)
    project_path = get_project_path(project_name)
    sessions = data['sessions']

    if len(sessions) == 0:
        print(f"No verified sessions found for project {project_name} with labeling {labeling}.")
        continue
    
    Xp,yp = make_windowed_dataset_from_sessions(sessions, window_size, window_stride, project_path)
    X.append(Xp)
    y.append(yp)

X = torch.cat(X)
y = torch.cat(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
torch.save((X_train,y_train),'train.pt')
torch.save((X_test,y_test),'test.pt')

print(f"Train dataset saved with shape: {X_train.shape}, {y_train.shape}")
print("Label distribution in train set:", y_train.long().bincount())
print(f"Test dataset saved with shape: {X_test.shape}, {y_test.shape}")
print("Label distribution in test set:", y_test.long().bincount())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmokingCNN(window_size=hyperparameters['window_size'], num_features=hyperparameters['num_features'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=hyperparameters['weight_decay'])
trainloader = DataLoader(TensorDataset(*torch.load(f'train.pt')), batch_size=hyperparameters['batch_size'], shuffle=True)
testloader = DataLoader(TensorDataset(*torch.load(f'test.pt')), batch_size=hyperparameters['batch_size'], shuffle=False)

trainlossi = []
testlossi = []
trainf1i = []
testf1i = []

n_epochs_without_improvement = 0
min_test_loss = float('inf')
max_test_f1 = 0.0
max_target_f1 = 0.0
experiment_name = hyperparameters['experiment_name']
os.makedirs(f'experiments/{experiment_name}', exist_ok=True)

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

        if testf1i[-1] > max_test_f1:
            max_test_f1 = testf1i[-1]
            torch.save(model.state_dict(), f'experiments/{experiment_name}/best_model_base_f1.pt')
            torch.save(model.state_dict(), f'/home/andrew/.delta/models/all.pt')
            print(f"Epoch {epoch}: New best model saved with test F1 score {max_test_f1:.4f}")
            n_epochs_without_improvement = 0
        else:
            n_epochs_without_improvement += 1

        if n_epochs_without_improvement >= hyperparameters['patience']:
            print(f"Early stopping triggered after {n_epochs_without_improvement} epochs without improvement.")
            n_epochs_without_improvement = 0
            break
            
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