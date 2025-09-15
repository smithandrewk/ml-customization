import os
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch import nn
from utils_simple import *
import torch
import matplotlib.pyplot as plt

# Hyperparameters
lr = 3e-4
batch_size = 32
early_stopping_patience = 500
window_size = 3000
num_features = 6
data_path = f'data/001_test'
target_participant = 'asfik'

new_exp_dir = create_and_get_new_exp_dir()
target_train_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_train.pt'))
target_val_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_val.pt'))
target_test_dataset = TensorDataset(*torch.load(f'{data_path}/{target_participant}_test.pt'))

print(f'Target train dataset size: {len(target_train_dataset)}')
print(f'Target val dataset size: {len(target_val_dataset)}')
print(f'Target test dataset size: {len(target_test_dataset)}')

from utils import SimpleSmokingCNN
model = SimpleSmokingCNN(window_size=window_size, num_features=num_features)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
metrics = {
    'best_val_loss': None,
    'best_val_loss_epoch': None,
}
best_val_loss = float('inf')
patience_counter = 0
device = 'cuda'

trainloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(target_val_dataset, batch_size=batch_size)
testloader = DataLoader(target_test_dataset, batch_size=batch_size)

lossi = {
    'train_loss': [],
    'val_loss': [],
    'test_loss': [],
    'train_f1': [],
    'val_f1': [],
    'test_f1': [],
}

model.to(device)
epoch = 0

from time import time

while True:
    start_time = time()
    model.train()
    train_loss, train_f1 = optimize_model_compute_loss_and_f1(model, trainloader, optimizer, criterion, device=device)

    lossi['train_loss'].append(train_loss)
    lossi['train_f1'].append(train_f1)

    loss,f1 = compute_loss_and_f1(model, valloader, criterion, device=device)
    lossi['val_loss'].append(loss)
    lossi['val_f1'].append(f1)

    loss,f1 = compute_loss_and_f1(model, testloader, criterion, device=device)
    lossi['test_loss'].append(loss)
    lossi['test_f1'].append(f1)

    if lossi['val_loss'][-1] < best_val_loss:
        best_val_loss = lossi['val_loss'][-1]
        metrics[f'best_val_loss'] = best_val_loss
        metrics[f'best_val_loss_epoch'] = epoch
        torch.save(model.state_dict(), f'{new_exp_dir}/best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered")
        torch.save(model.state_dict(), f'{new_exp_dir}/last_model.pt')
        break

    plt.figure(figsize=(7.2,4.48))
    plt.plot(lossi['train_loss'], label='Train Loss (base)')
    plt.plot(lossi['val_loss'], label='Val Loss (target)', color='g')
    plt.plot(lossi['test_loss'], label='Test Loss (target)', color='r')

    if metrics['best_val_loss_epoch'] is not None and metrics['best_val_loss'] is not None:
        plt.axhline(y=metrics['best_val_loss'], color='b', linestyle='--', label='Best Base Val Loss', alpha=0.5)
        plt.axvline(x=metrics['best_val_loss_epoch'], color='b', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/loss.jpg', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7.2,4.48))
    plt.plot(lossi['train_f1'], label='Train f1 (base)')
    plt.plot(lossi['val_f1'], label='Val f1 (target)', color='g')
    plt.plot(lossi['test_f1'], label='Test f1 (target)', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([.7, 1])
    plt.legend()
    plt.title(f'Patience Counter: {patience_counter}')
    plt.savefig(f'{new_exp_dir}/f1.jpg', bbox_inches='tight')
    plt.close()

    epoch += 1
    print(f'Epoch {epoch}, Time Elapsed: {time() - start_time:.2f}s')
