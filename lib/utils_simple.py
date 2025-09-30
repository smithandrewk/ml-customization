import torch
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from torch import nn
import matplotlib.pyplot as plt
import math
from torch.nn.functional import relu
import os

class EarlyStopper:
    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded
    
def plot_loss_and_f1(lossi, new_exp_dir, metrics, patience_counter):
    plt.figure(figsize=(7.2,4.48),dpi=300)
    plt.plot(lossi['base train loss'], label='Train Loss (base)', color='b')
    plt.plot(lossi['base val loss'], label='Val Loss (base)', color='b', linestyle='--')
    plt.plot(lossi['target train loss'], label='Train Loss (target)', color='g')
    plt.plot(lossi['target val loss'], label='Val Loss (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_loss_epoch'] is not None and metrics['best_base_val_loss'] is not None:
        plt.axhline(y=metrics['best_base_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val loss'][metrics['best_base_val_loss_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_loss_epoch'] is not None and metrics['best_target_val_loss'] is not None:
        plt.axhline(y=metrics['best_target_val_loss'], color='g', linestyle='--', label='Best target Val Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_loss_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.savefig(f'loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48),dpi=300)
    plt.plot(lossi['base train f1'], label='Train f1 (base)', color='b')
    plt.plot(lossi['base val f1'], label='Val f1 (base)', color='b', linestyle='--')
    plt.plot(lossi['target train f1'], label='Train f1 (target)', color='g')
    plt.plot(lossi['target val f1'], label='Val f1 (target)', color='g', linestyle='--')

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_f1_epoch'] is not None and metrics['best_base_val_f1'] is not None:
        plt.axhline(y=metrics['best_base_val_f1'], color='b', linestyle='--', label='Best Base f1 Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_f1_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val f1'][metrics['best_base_val_f1_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_f1_epoch'] is not None and metrics['best_target_val_f1'] is not None:
        plt.axhline(y=metrics['best_target_val_f1'], color='g', linestyle='--', label='Best target f1 Loss', alpha=0.8)
        plt.axvline(x=metrics['best_target_val_f1_epoch'], color='g', linestyle='--', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.savefig(f'f1.jpg', bbox_inches='tight')
    plt.close()

def create_and_get_new_exp_dir(prefix='exp'):
    # Create experiments directory if it doesn't exist
    base_exp_dir = f'experiments/{prefix}'
    os.makedirs(base_exp_dir, exist_ok=True)

    # Find the highest existing experiment number in the prefix directory
    max_exp_num = 0
    if os.path.exists(base_exp_dir):
        for dirname in os.listdir(base_exp_dir):
            if os.path.isdir(os.path.join(base_exp_dir, dirname)):
                try:
                    exp_num = int(dirname)
                    max_exp_num = max(max_exp_num, exp_num)
                except ValueError:
                    continue

    # Create new experiment directory with zero-padded integer
    new_exp_num = max_exp_num + 1
    new_exp_dir = f'{base_exp_dir}/{new_exp_num:04d}'
    os.makedirs(new_exp_dir, exist_ok=True)
    return new_exp_dir

def plot_loss(patience_counter, lossi, evalloaders, metrics, exp_dir):
    plt.figure(figsize=(7.2,4.48))

    plt.plot(lossi['train loss'], label='Train Loss (base)')
    plt.plot(lossi['target train loss'], label='Train Loss (target)', linestyle='-',color='g')

    for evalloader in evalloaders:
        name = evalloader[1]
        plt.plot(lossi[f'{name} loss'], label=f'{name} loss', color=evalloader[2])
        plt.plot(lossi[f'{name} f1'], label=f'{name} f1', color=evalloader[2])

    if metrics['transition_epoch'] is not None:
        plt.axvline(x=metrics['transition_epoch'], color='r', linestyle='--', label='Phase Transition')

    if metrics['best_base_val_loss_epoch'] is not None and metrics['best_base_val_loss'] is not None:
        plt.axhline(y=metrics['best_base_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_base_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)
        plt.axhline(y=lossi['target val loss'][metrics['best_base_val_loss_epoch']], color='g', linestyle='--', label='Best Base Val Loss', alpha=0.5)

    if metrics['best_target_val_loss_epoch'] is not None and metrics['best_target_val_loss'] is not None:
        plt.axhline(y=metrics['best_target_val_loss'], color='g', linestyle='--', label='Best target Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_target_val_loss_epoch'], color='g', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{exp_dir}/loss.jpg', bbox_inches='tight')
    plt.close()

def compute_loss(model, dataloader, criterion, device, log=False):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for Xi, yi in dataloader:
            Xi = Xi.to(device)
            yi = yi.to(device).float()
            logits = model(Xi).squeeze()
            loss = criterion(logits, yi.squeeze())
            total_loss += loss.item() * Xi.size(0)
            count += Xi.size(0)
    if log:
        return math.log(total_loss / count)
    return total_loss / count

from sklearn.metrics import f1_score

def compute_loss_and_f1(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for Xi, yi in dataloader:
            Xi = Xi.to(device)
            yi = yi.to(device).float()
            logits = model(Xi).squeeze(-1)
            loss = criterion(logits, yi)
            total_loss += loss.item() * Xi.size(0)
            count += Xi.size(0)
            y_true.append(yi.cpu())
            y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    f1 = f1_score(y_true, y_pred, average='macro')
    return total_loss / count, float(f1)

def optimize_model_and_compute_loss(model, dataloader, optimizer, criterion, device, log=False):
    model.train()
    total_loss = 0.0
    count = 0
    for Xi, yi in dataloader:
        Xi = Xi.to(device)
        yi = yi.to(device).float()
        optimizer.zero_grad()
        logits = model(Xi).squeeze()
        loss = criterion(logits, yi.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xi.size(0)
        count += Xi.size(0)
    if log:
        return math.log(total_loss / count)
    return total_loss / count

def optimize_model_compute_loss_and_f1(model, dataloader, optimizer, criterion, device, augmenter=None):
    model.train()
    total_loss = 0.0
    count = 0
    y_true = []
    y_pred = []
    for Xi, yi in dataloader:
        Xi = Xi.to(device)
        yi = yi.to(device).float().view(-1,1)

        # Apply augmentation if provided
        if augmenter is not None:
            Xi, yi = augmenter.augment(Xi, yi)

        optimizer.zero_grad()
        logits = model(Xi)
        loss = criterion(logits,yi)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xi.size(0)
        count += Xi.size(0)
        y_true.append(yi.cpu())
        y_pred.append(logits.sigmoid().round().cpu())
    y_true = torch.cat(y_true).cpu().detach()
    y_pred = torch.cat(y_pred).cpu().detach()
    f1 = f1_score(y_true, y_pred, average='macro')
    # f1 = (2 * (y_true * y_pred).sum()) / ((y_true + y_pred).sum() + 1e-8)
    return total_loss / count, float(f1)

def evaluate(model, dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for Xi,yi in dataloader:
            Xi = Xi.to(device)
            y_true.append(yi)
            y_pred.append(model(Xi).sigmoid().round().cpu().flatten())
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()

    print(classification_report(y_true, y_pred, target_names=['No Smoking', 'Smoking']))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,normalize='true')
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,normalize='pred')
    return y_true,y_pred

class SimpleCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c1 = nn.Conv1d(6, 16, kernel_size=5, padding=1)
        self.mp1 = nn.MaxPool1d(2)
        self.c2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.c1(x)
        x = relu(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    