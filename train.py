"""
Train test split across sessions for multiple participants
"""
import os
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from sklearn.metrics import f1_score

from utils import load_config, get_experiment_dir, plot_training_progress, SmokingCNN
from dotenv import load_dotenv

load_dotenv()

# Load configuration
config = load_config()
experiment_dir = get_experiment_dir(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(experiment_dir, exist_ok=True)

model = SmokingCNN(
    window_size=config['model']['window_size'], 
    num_features=config['model']['num_features']
)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(
    model.parameters(), 
    lr=config['training']['learning_rate'], 
    weight_decay=config['training']['weight_decay']
)

trainloader = DataLoader(
    TensorDataset(*torch.load(f'{experiment_dir}/train.pt')), 
    batch_size=config['training']['batch_size'], 
    shuffle=True
)
testloader = DataLoader(
    TensorDataset(*torch.load(f'{experiment_dir}/test.pt')), 
    batch_size=config['training']['batch_size'], 
    shuffle=False
)

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

for epoch in range(config['training']['num_epochs']):
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
                ma_window_size=config['visualization']['ma_window_size'],
                save_path=f'{experiment_dir}/loss.{config["visualization"]["plot_format"]}',
            )

        ## Early Stopping Logic on F1 Score
        if testf1i[-1] > max_test_f1:
            max_test_f1 = testf1i[-1]
            torch.save(model.state_dict(), f'{experiment_dir}/best_model_base_f1.pt')
            print(f"Epoch {epoch}: New best model saved with test F1 score {max_test_f1:.4f}")
            n_epochs_without_improvement = 0
        else:
            print(f"Epoch {epoch}: No improvement in test F1 score ({testf1i[-1]:.4f} vs {max_test_f1:.4f})")
            n_epochs_without_improvement += 1
        if n_epochs_without_improvement >= config['training']['patience']:
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
print(f"Final model saved to: {experiment_dir}/last.pt")

torch.save(model.state_dict(), f'{experiment_dir}/last.pt')

metrics = {
    'train_loss': trainlossi,
    'test_loss': testlossi,
    'train_f1': trainf1i,
    'test_f1': testf1i,
    'config': config,
    'best_f1': max_test_f1,
    'train_samples': train_dataset_size,
    'test_samples': test_dataset_size,
    'total_samples': train_dataset_size + test_dataset_size
}

torch.save(metrics, f'{experiment_dir}/metrics.pt')
