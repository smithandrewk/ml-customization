import mysql.connector
from mysql.connector import Error
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from contextlib import contextmanager
import yaml

load_dotenv()

"""
Configuration and database utils
"""

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Calculate derived parameters
    config['dataset']['window_size'] = config['dataset']['fs'] * config['dataset']['window_size_seconds']
    config['dataset']['window_stride'] = config['dataset']['fs'] * config['dataset']['window_stride_seconds']
    
    # Update model window_size to match dataset
    config['model']['window_size'] = config['dataset']['window_size']
    
    return config

def get_experiment_dir(config):
    """Get the experiment directory path."""
    return os.path.join(config['experiment']['save_dir'], config['experiment']['name'])

@contextmanager
def db_connection():
    """Context manager for database connections that ensures proper cleanup."""
    conn = None
    try:
        MYSQL_CONFIG = {
            'host': os.getenv('MYSQL_HOST', '10.173.98.204'),
            'user': os.getenv('MYSQL_USER', 'andrew'),
            'password': os.getenv('MYSQL_PASSWORD', 'admin'),
            'database': os.getenv('MYSQL_DATABASE', 'delta2')
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        yield conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_db_connection():
    try:
        MYSQL_CONFIG = {
            'host': '10.173.98.204',
            'user': 'andrew',
            'password': 'admin',
            'database': 'delta2'
        }
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None
    
def moving_average(data, window_size):
    """Calculate moving average with specified window size."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

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

def load_data(raw_dataset_path, session_name, start_ns=None, stop_ns=None):
    """Load accelerometer data for a specific session, optionally filtered by time range"""
    try:
        
        # Try exact match first
        session_path = os.path.join(raw_dataset_path, session_name.split('.')[0])
        
        print(f"Trying session path: {session_path}")
        accelerometer_path = os.path.join(session_path, 'accelerometer_data.csv')
        
        # Load using n_rows and offset with start_ns and stop_ns if provided
        df = pd.read_csv(accelerometer_path)
        if 'accel_x' not in df.columns:
            df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
        
        # Filter by time range if provided
        if start_ns is not None and stop_ns is not None:
            mask = (df['ns_since_reboot'] >= start_ns) & (df['ns_since_reboot'] <= stop_ns)
            df = df[mask].copy()
            df.reset_index(drop=True, inplace=True)
        
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data for session {session_name}: {e}")
        return None
    
def make_windowed_dataset_from_sessions(sessions, window_size, window_stride, raw_dataset_path, labeling='andrew smoking labels'):
    X = []
    y = []

    for session in sessions:
        session_name = session['session_name']
        start_ns = session.get('start_ns')
        stop_ns = session.get('stop_ns')
        bouts = [b for b in session['bouts'] if b['label'] == labeling]

        df = load_data(raw_dataset_path, session_name, start_ns, stop_ns)
        df['label'] = 0

        for bout in bouts:
            start = bout['start']
            end = bout['end']
            df.loc[(df['ns_since_reboot'] >= start) & (df['ns_since_reboot'] <= end), 'label'] = 1

        if 'accel_x' not in df.columns or 'accel_y' not in df.columns or 'accel_z' not in df.columns:
            df.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)

        data = torch.tensor(df[['accel_x', 'accel_y', 'accel_z','label']].values, dtype=torch.float32)

        if data.shape[0] < window_size:
            # Zero pad the data to window size
            padding_length = window_size - data.shape[0]
            padding = torch.zeros((padding_length, data.shape[1]), dtype=torch.float32)
            data = torch.cat([data, padding], dim=0)
            print(f"Zero-padded session {session_name} from {data.shape[0] - padding_length} to {data.shape[0]} samples")

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
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT path FROM projects WHERE project_name = %s"
        cursor.execute(query, (project_name,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_project_id(project_name):
    """Get the project ID from the database."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_id FROM projects WHERE project_name = %s"
        cursor.execute(query, (project_name,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_raw_dataset_path(project_name):
    """Get the raw dataset path for a project."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get project_id
        project_id = get_project_id(project_name)
        
        # Get dataset_id from project_dataset_refs
        query = "SELECT dataset_id FROM project_dataset_refs WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        ref_row = cursor.fetchone()
        
        if not ref_row:
            raise ValueError(f"No raw dataset reference found for project '{project_name}'")
        
        dataset_id = ref_row[0]
        
        # Get file_path from raw_datasets table
        query = "SELECT file_path FROM raw_datasets WHERE dataset_id = %s"
        cursor.execute(query, (dataset_id,))
        path_row = cursor.fetchone()
    
    if path_row and path_row[0]:
        return path_row[0]
    else:
        raise ValueError(f"No file_path found for dataset {dataset_id}")

def get_sessions_for_project(project_name):
    """Get all sessions for a project."""
    with db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        project_id = get_project_id(project_name)
        query = "SELECT * FROM sessions WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        rows = cursor.fetchall()
    
    for row in rows:
        if 'bouts' in row and row['bouts']:
            try:
                row['bouts'] = json.loads(row['bouts'])
            except json.JSONDecodeError:
                row['bouts'] = []
    
    return rows
    

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
    """Get the participant ID from the database based on participant code."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT participant_id FROM participants WHERE participant_code = %s"
        cursor.execute(query, (participant_code,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Participant code '{participant_code}' not found in the database.")
    
def get_participant_projects(participant_id):
    """Get all projects associated with a participant."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_name FROM projects WHERE participant_id = %s"
        cursor.execute(query, (participant_id,))
        rows = cursor.fetchall()
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

def generate_dataset_summary(config, session_data, train_test_splits, X_train, y_train, X_test, y_test, save_dir):
    """
    Generate comprehensive dataset summary with statistics and save as CSV and text report.
    
    Args:
        config: Configuration dictionary
        session_data: Dictionary mapping project_name -> list of sessions
        train_test_splits: Dictionary mapping project_name -> (train_sessions, test_sessions)
        X_train, y_train: Training data tensors
        X_test, y_test: Test data tensors
        save_dir: Directory to save summary files
    """
    import pandas as pd
    from datetime import datetime
    import os
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Collect session-level statistics
    session_stats = []
    project_stats = []
    
    for project_name, sessions in session_data.items():
        train_sessions, test_sessions = train_test_splits[project_name]
        train_session_names = {s['session_name'] for s in train_sessions}
        
        # Project-level aggregates
        total_sessions = len(sessions)
        total_duration = 0
        total_smoking_duration = 0
        smoking_sessions = 0
        
        for session in sessions:
            session_name = session['session_name']
            split_type = 'train' if session_name in train_session_names else 'test'
            
            # Calculate session duration (convert from nanoseconds to minutes)
            if session.get('start_ns') and session.get('stop_ns'):
                duration_minutes = (session['stop_ns'] - session['start_ns']) * 1e-9 / 60
            else:
                duration_minutes = 0
            
            # Calculate smoking duration in this session
            smoking_duration_minutes = 0
            smoking_bouts = [b for b in session.get('bouts', []) if b.get('label') == config['dataset']['labeling']]
            for bout in smoking_bouts:
                smoking_duration_minutes += (bout['end'] - bout['start']) * 1e-9 / 60
            
            if smoking_duration_minutes > 0:
                smoking_sessions += 1
            
            total_duration += duration_minutes
            total_smoking_duration += smoking_duration_minutes
            
            session_stats.append({
                'project_name': project_name,
                'session_name': session_name,
                'split': split_type,
                'duration_minutes': round(duration_minutes, 2),
                'smoking_duration_minutes': round(smoking_duration_minutes, 2),
                'smoking_percentage': round((smoking_duration_minutes / duration_minutes * 100) if duration_minutes > 0 else 0, 2),
                'num_smoking_bouts': len(smoking_bouts),
                'keep_status': session.get('keep', 1)
            })
        
        project_stats.append({
            'project_name': project_name,
            'total_sessions': total_sessions,
            'train_sessions': len(train_sessions),
            'test_sessions': len(test_sessions),
            'total_duration_hours': round(total_duration / 60, 2),
            'smoking_duration_hours': round(total_smoking_duration / 60, 2),
            'smoking_percentage': round((total_smoking_duration / total_duration * 100) if total_duration > 0 else 0, 2),
            'sessions_with_smoking': smoking_sessions,
            'smoking_session_percentage': round((smoking_sessions / total_sessions * 100) if total_sessions > 0 else 0, 2)
        })
    
    # Convert to DataFrames
    session_df = pd.DataFrame(session_stats)
    project_df = pd.DataFrame(project_stats)
    
    # Calculate windowed data statistics
    train_samples = len(X_train)
    test_samples = len(X_test)
    total_samples = train_samples + test_samples
    
    train_positive = int(torch.sum(y_train).item())
    train_negative = int(train_samples - train_positive)
    test_positive = int(torch.sum(y_test).item())
    test_negative = int(test_samples - test_positive)
    
    # Create summary statistics
    summary_stats = {
        'dataset_info': {
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config_file': 'config.yaml',
            'experiment_name': config['experiment']['name'],
            'labeling_type': config['dataset']['labeling'],
            'window_size_samples': config['dataset']['window_size'],
            'window_size_seconds': config['dataset']['window_size_seconds'],
            'window_stride_samples': config['dataset']['window_stride'],
            'window_stride_seconds': config['dataset']['window_stride_seconds'],
            'sampling_frequency_hz': config['dataset']['fs'],
            'test_split_ratio': config['dataset']['test_size']
        },
        'project_summary': {
            'total_projects': len(project_stats),
            'total_sessions': session_df.shape[0],
            'total_duration_hours': round(session_df['duration_minutes'].sum() / 60, 2),
            'total_smoking_hours': round(session_df['smoking_duration_minutes'].sum() / 60, 2),
            'overall_smoking_percentage': round((session_df['smoking_duration_minutes'].sum() / session_df['duration_minutes'].sum() * 100), 2)
        },
        'split_summary': {
            'train_sessions': len(session_df[session_df['split'] == 'train']),
            'test_sessions': len(session_df[session_df['split'] == 'test']),
            'train_duration_hours': round(session_df[session_df['split'] == 'train']['duration_minutes'].sum() / 60, 2),
            'test_duration_hours': round(session_df[session_df['split'] == 'test']['duration_minutes'].sum() / 60, 2)
        },
        'windowed_data_summary': {
            'total_windows': total_samples,
            'train_windows': train_samples,
            'test_windows': test_samples,
            'train_percentage': round((train_samples / total_samples * 100), 2),
            'test_percentage': round((test_samples / total_samples * 100), 2),
            'train_positive_windows': train_positive,
            'train_negative_windows': train_negative,
            'test_positive_windows': test_positive,
            'test_negative_windows': test_negative,
            'train_positive_percentage': round((train_positive / train_samples * 100), 2),
            'train_negative_percentage': round((train_negative / train_samples * 100), 2),
            'test_positive_percentage': round((test_positive / test_samples * 100), 2),
            'test_negative_percentage': round((test_negative / test_samples * 100), 2),
            'overall_positive_percentage': round(((train_positive + test_positive) / total_samples * 100), 2),
            'class_balance_ratio': round((train_negative + test_negative) / (train_positive + test_positive), 2)
        }
    }
    
    # Save session-level details as CSV
    session_csv_path = os.path.join(save_dir, f'dataset_sessions_{timestamp}.csv')
    session_df.to_csv(session_csv_path, index=False)
    
    # Save project-level summary as CSV
    project_csv_path = os.path.join(save_dir, f'dataset_projects_{timestamp}.csv')
    project_df.to_csv(project_csv_path, index=False)
    
    # Save comprehensive text summary
    summary_txt_path = os.path.join(save_dir, f'dataset_summary_{timestamp}.txt')
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET CONFIGURATION\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['dataset_info'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nPROJECT OVERVIEW\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['project_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nTRAIN/TEST SPLIT OVERVIEW\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['split_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nWINDOWED DATA STATISTICS\n")
        f.write("-"*40 + "\n")
        for key, value in summary_stats['windowed_data_summary'].items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        f.write("\nPROJECT DETAILS\n")
        f.write("-"*40 + "\n")
        f.write(project_df.to_string(index=False))
        
        f.write("\n\nCLASS DISTRIBUTION BY SPLIT\n")
        f.write("-"*40 + "\n")
        f.write(f"Training Set:\n")
        f.write(f"  - Negative (No Smoking): {train_negative:,} windows ({summary_stats['windowed_data_summary']['train_negative_percentage']:.1f}%)\n")
        f.write(f"  - Positive (Smoking): {train_positive:,} windows ({summary_stats['windowed_data_summary']['train_positive_percentage']:.1f}%)\n")
        f.write(f"\nTest Set:\n")
        f.write(f"  - Negative (No Smoking): {test_negative:,} windows ({summary_stats['windowed_data_summary']['test_negative_percentage']:.1f}%)\n")
        f.write(f"  - Positive (Smoking): {test_positive:,} windows ({summary_stats['windowed_data_summary']['test_positive_percentage']:.1f}%)\n")
        
        f.write(f"\nOverall Class Balance:\n")
        f.write(f"  - Negative:Positive Ratio = {summary_stats['windowed_data_summary']['class_balance_ratio']:.2f}:1\n")
        
    print(f"\n📊 Dataset summary generated:")
    print(f"   📄 Session details: {session_csv_path}")
    print(f"   📈 Project summary: {project_csv_path}")
    print(f"   📋 Full report: {summary_txt_path}")
    
    return {
        'session_df': session_df,
        'project_df': project_df,
        'summary_stats': summary_stats,
        'files': {
            'sessions_csv': session_csv_path,
            'projects_csv': project_csv_path,
            'summary_txt': summary_txt_path
        }
    }