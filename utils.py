import mysql.connector
from mysql.connector import Error
import os
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
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def moving_average(data, window_size):
    """Calculate moving average with specified window size."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def get_next_experiment_number():
    experiments = os.listdir('experiments')
    if len(experiments) == 0:
        return 1
    else:
        return max([int(exp.split('_')[0]) for exp in experiments if os.path.isdir(os.path.join('experiments', exp))]) + 1
    
def plot_training_progress(trainlossi, testlossi, targetlossi=None, trainf1i=None, testf1i=None, targetf1i=None, ma_window_size=10, save_path='training_metrics.jpg',transition_epoch=None):
    """
    Plot training progress with loss on top subplot and F1 scores on bottom subplot.
    Both subplots share the x-axis.
    """
    # Set up figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    
    # Setup colors with professional palette
    colors = {
        'train': '#0072B2',  # Professional blue
        'test': '#D55E00',    # Professional orange
        'target': '#009E73'  # Professional green
    }
    
    # PLOT 1: LOSS VALUES
    # Plot raw training loss with low opacity
    ax1.plot(np.linspace(0, len(testlossi)-1, len(trainlossi)), trainlossi, 
             alpha=0.3, color=colors['train'], linewidth=1, label='_nolegend_')
    
    ax1.plot(np.linspace(0, len(testlossi)-1, len(testlossi)), testlossi, 
             alpha=0.3, color=colors['test'], linewidth=1, label='_nolegend_')
    
    if targetlossi is not None:
        ax1.plot(np.linspace(0, len(testlossi)-1, len(targetlossi)), targetlossi, 
                alpha=0.3, color=colors['target'], linewidth=1, label='_nolegend_')
    
    # Plot moving average if we have enough data
    if len(trainlossi) > ma_window_size:
        trainlossi_ma = moving_average(trainlossi, ma_window_size)
        x_trainlossi_ma = np.linspace(ma_window_size-1, len(testlossi)-1, len(trainlossi_ma))
        ax1.plot(x_trainlossi_ma, trainlossi_ma, color=colors['train'], 
                 linewidth=2, label='Training Loss (MA)')
        
        # Mark minimum training loss
        min_idx = np.argmin(trainlossi_ma)
        min_val = np.min(trainlossi_ma)
        ax1.plot(x_trainlossi_ma[min_idx], min_val, 'o', color=colors['train'], markersize=8)
        ax1.annotate(f'{min_val:.4f}', xy=(x_trainlossi_ma[min_idx], min_val),
                    xytext=(5, -15), textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color=colors['train']))
    # Plot moving average if we have enough data
    if len(testlossi) > ma_window_size:
        testlossi_ma = moving_average(testlossi, ma_window_size)
        x_testlossi_ma = np.linspace(ma_window_size-1, len(testlossi)-1, len(testlossi_ma))
        ax1.plot(x_testlossi_ma, testlossi_ma, color=colors['test'], 
                 linewidth=2, label='Test Loss (MA)')
        
        # Mark minimum testing loss
        min_idx = np.argmin(testlossi_ma)
        min_val = np.min(testlossi_ma)
        ax1.plot(x_testlossi_ma[min_idx], min_val, 'o', color=colors['test'], markersize=8)
        ax1.annotate(f'{min_val:.4f}', xy=(x_testlossi_ma[min_idx], min_val),
                    xytext=(5, -15), textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color=colors['test']))
    
    if targetlossi is not None:
        # Plot moving average if we have enough data
        if len(targetlossi) > ma_window_size:
            targetlossi_ma = moving_average(targetlossi, ma_window_size)
            x_targetlossi_ma = np.linspace(ma_window_size-1, len(testlossi)-1, len(targetlossi_ma))
            ax1.plot(x_targetlossi_ma, targetlossi_ma, color=colors['target'], 
                    linewidth=2, label='Target Loss (MA)')
            
            # Mark minimum targeting loss
            min_idx = np.argmin(targetlossi_ma)
            min_val = np.min(targetlossi_ma)
            ax1.plot(x_targetlossi_ma[min_idx], min_val, 'o', color=colors['target'], markersize=8)
            ax1.annotate(f'{min_val:.4f}', xy=(x_targetlossi_ma[min_idx], min_val),
                        xytext=(5, -15), textcoords='offset points', fontsize=10,
                        arrowprops=dict(arrowstyle='->', color=colors['target']))
    
    # Professional styling for loss subplot
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold', pad=10)
    
    # Only show legend if there are labeled plots
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(frameon=True, framealpha=0.9)
    
    # PLOT 2: F1 SCORES (only if provided)
    # Plot training F1 scores
    x_trainf1i = np.array(range(len(trainf1i)))
    ax2.plot(x_trainf1i, trainf1i, '-', color=colors['train'], 
            linewidth=2,alpha=0.3)
    
    # Plot validation F1 scores
    x_testf1i = np.array(range(len(testf1i)))
    ax2.plot(x_testf1i, testf1i, 'o-', color=colors['test'], 
            linewidth=2,alpha=0.3)
    
    if targetf1i is not None:
        # Plot validation F1 scores
        x_targetf1i = np.array(range(len(targetf1i)))
        ax2.plot(x_targetf1i, targetf1i, 'o-', color=colors['target'], 
                linewidth=2,alpha=0.3)
    
    # Plot moving average if we have enough data
    if len(trainf1i) > ma_window_size:
        trainf1i_ma = moving_average(trainf1i, ma_window_size)
        x_trainf1i_ma = np.linspace(ma_window_size-1, len(testf1i)-1, len(trainf1i_ma))
        ax2.plot(x_trainf1i_ma, trainf1i_ma, color=colors['train'], 
                linewidth=2, label='Train F1 (MA)')
        
        # Mark minimum training loss
        max_idx = np.argmax(trainf1i_ma)
        max_val = np.max(trainf1i_ma)
        ax2.plot(x_trainf1i_ma[max_idx], max_val, 'o', color=colors['train'], markersize=8)
        ax2.annotate(f'{max_val:.4f}', xy=(x_trainf1i_ma[max_idx], max_val),
                    xytext=(5, -15), textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color=colors['train']))
        

    # Plot moving average if we have enough data
    if len(testf1i) > ma_window_size:
        testf1i_ma = moving_average(testf1i, ma_window_size)
        x_testf1i_ma = np.linspace(ma_window_size-1, len(testf1i)-1, len(testf1i_ma))
        ax2.plot(x_testf1i_ma, testf1i_ma, color=colors['test'], 
                linewidth=2, label='Test F1 (MA)')
        
        # Mark minimum testing loss
        max_idx = np.argmax(testf1i_ma)
        max_val = np.max(testf1i_ma)
        ax2.plot(x_testf1i_ma[max_idx], max_val, 'o', color=colors['test'], markersize=8)
        ax2.annotate(f'{max_val:.4f}', xy=(x_testf1i_ma[max_idx], max_val),
                    xytext=(5, -15), textcoords='offset points', fontsize=10,
                    arrowprops=dict(arrowstyle='->', color=colors['test']))
    if targetf1i is not None:
        # Plot moving average if we have enough data
        if len(targetf1i) > ma_window_size:
            targetf1i_ma = moving_average(targetf1i, ma_window_size)
            x_targetf1i_ma = np.linspace(ma_window_size-1, len(testf1i)-1, len(targetf1i_ma))
            ax2.plot(x_targetf1i_ma, targetf1i_ma, color=colors['target'], 
                    linewidth=2, label='Target F1 (MA)')
            
            # Mark minimum targeting loss
            max_idx = np.argmax(targetf1i_ma)
            max_val = np.max(targetf1i_ma)
            ax2.plot(x_targetf1i_ma[max_idx], max_val, 'o', color=colors['target'], markersize=8)
            ax2.annotate(f'{max_val:.4f}', xy=(x_targetf1i_ma[max_idx], max_val),
                        xytext=(5, -15), textcoords='offset points', fontsize=10,
                        arrowprops=dict(arrowstyle='->', color=colors['target']))
            
    if transition_epoch is not None:
        # Add vertical line for transition epoch
        if transition_epoch > 0:
            # Add vertical line for transition epoch
            ax1.axvline(x=transition_epoch, color='gray', linestyle='--', linewidth=1, label='Transition Epoch')
            ax2.axvline(x=transition_epoch, color='gray', linestyle='--', linewidth=1, label='Transition Epoch')

    # Professional styling for F1 subplot
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Scores', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim([0, 1.05])  # F1 scores are between 0 and 1
    
    # Only show legend if there are labeled plots
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(frameon=True, framealpha=0.9)


        
    # Main title for the whole figure
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.12)  # Reduce space between subplots
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def make_windowed_dataset_from_sessions(sessions, window_size, window_stride, project_path):
    X = []
    y = []

    for train_session in sessions:
        session_name = train_session['session_name']
        session_path = os.path.join(project_path, session_name)
        bouts = train_session['bouts']
        accelerometer_path = os.path.join(session_path, 'accelerometer_data.csv')
        accelerometer_df = pd.read_csv(accelerometer_path)

        accelerometer_df['label'] = 0

        for bout in bouts:
            start = bout['start']
            end = bout['end']
            accelerometer_df.loc[(accelerometer_df['ns_since_reboot'] >= start) & (accelerometer_df['ns_since_reboot'] <= end), 'label'] = 1

        if 'accel_x' not in accelerometer_df.columns or 'accel_y' not in accelerometer_df.columns or 'accel_z' not in accelerometer_df.columns:
            # rename columns to match expected names
            accelerometer_df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
        data = torch.tensor(accelerometer_df[['accel_x', 'accel_y', 'accel_z','label']].values, dtype=torch.float32)

        if data.shape[0] < window_size:
            # TODO: zero pad the data to window size
            print(f"Skipping session {session_name} due to insufficient data length: {data.shape[0]} < {window_size}")
            continue

        windowed_data = data.unfold(dimension=0,size=window_size,step=window_stride)

        X.append(windowed_data[:,:-1,:])
        y.append(windowed_data[:,-1,:])

    X = torch.cat(X)
    y = (~(torch.cat(y) == 0).all(axis=1)).float()
    return X,y

def get_db_connection():
    try:
        MYSQL_CONFIG = {
            'host': os.getenv('MYSQL_HOST'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DATABASE')
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")

def get_project_path(project_name):
    """
    Get the project path from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute a query
    query = f"SELECT path FROM projects WHERE project_name='{project_name}'"  # Replace with your actual table name
    cursor.execute(query)
    # Fetch the first row from the executed query
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_project_id(project_name):
    """
    Get the project ID from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute a query
    query = f"SELECT project_id FROM projects WHERE project_name='{project_name}'"  # Replace with your actual table name
    cursor.execute(query)
    # Fetch the first row from the executed query
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")
    

def get_verified_and_not_deleted_sessions(project_name, labeling):
    """
    Get sessions from the database for a specific project and labeling.
    """
    # Connect to the database and fetch sessions
    conn = get_db_connection()
    cursor = conn.cursor()
    # Execute a query
    query = f"SELECT session_name,bouts,start_ns,stop_ns FROM sessions WHERE project_id='{get_project_id(project_name)}' AND verified=1 AND (keep IS NULL OR keep != 0)"  # Replace with your actual table name
    cursor.execute(query)
    # Fetch all rows from the executed query
    rows = cursor.fetchall()
    conn.close()
    data = {'sessions': 
            [
                {
                    'session_name': row[0], 
                    'bouts': [b for b in json.loads(row[1]) if type(b) == dict and b.get('label') == labeling],
                    'bout_duration': sum((bout['end'] - bout['start']) * 1e-9 for bout in json.loads(row[1]) if type(bout) == dict and bout.get('label') == labeling),
                    'session_duration': (row[3] - row[2]) * 1e-9
                } for row in rows
            ]}

    return data

def get_participant_id(participant_code):
    """
    Get the participant ID from the database based on participant code.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = f"SELECT participant_id FROM participants WHERE participant_code='{participant_code}'"
    cursor.execute(query)
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    else:
        raise ValueError(f"Participant code '{participant_code}' not found in the database.")
    
def get_participant_projects(participant_id):
    """
    Get all projects associated with a participant.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    query = f"SELECT project_name FROM projects WHERE participant_id={participant_id}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate(model, dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
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

from torch.nn.functional import relu
import torch.nn as nn

class SmokingCNN(nn.Module):
    def __init__(self, window_size=100, num_features=6):
        super(SmokingCNN, self).__init__()
        
        # Use larger kernel sizes and dilated convolutions for much larger receptive field
        self.c1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce sequence length by half
        
        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce sequence length by half again
        
        # Dilated convolutions to capture long-range dependencies
        self.c3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=6, dilation=2)
        self.c4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=12, dilation=4)
        self.c5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=24, dilation=8)
        
        # Additional layers for even more context
        self.c6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, padding=2)
        self.c7 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # First conv block with pooling
        x = relu(self.c1(x))
        x = self.pool1(x)
        
        x = relu(self.c2(x))
        x = self.pool2(x)
        
        # Dilated convolutions for long-range dependencies
        x = relu(self.c3(x))
        x = relu(self.c4(x))
        x = relu(self.c5(x))
        
        # Additional processing
        x = relu(self.c6(x))
        x = relu(self.c7(x))
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.classifier(x)
        
        return x
    
def get_projects_from_participant_codes(participant_codes):
    projects = []
    for participant_code in participant_codes:
        participant_id = get_participant_id(participant_code)
        participant_projects = get_participant_projects(participant_id)
        if len(participant_projects) == 0:
            print(f"No projects found for participant {participant_code}.")
            continue
        print(f"Participant {participant_code} has projects: {participant_projects}")
        projects.extend(participant_projects)
    return projects