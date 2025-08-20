import json
import mysql.connector
from mysql.connector import Error
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def get_project_id(project_name):
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    query = "SELECT project_id FROM projects WHERE project_name = %s"
    cursor.execute(query, (project_name,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_raw_dataset_path(project_name):
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    
    # Get project_id
    project_id = get_project_id(project_name)
    
    # Get dataset_id from project_dataset_refs
    query = """
    SELECT dataset_id FROM project_dataset_refs 
    WHERE project_id = %s
    """
    cursor.execute(query, (project_id,))
    ref_row = cursor.fetchone()
    
    if not ref_row:
        conn.close()
        raise ValueError(f"No raw dataset reference found for project '{project_name}'")
    
    dataset_id = ref_row[0]
    
    # Get file_path from raw_datasets table
    query = "SELECT file_path FROM raw_datasets WHERE dataset_id = %s"
    cursor.execute(query, (dataset_id,))
    path_row = cursor.fetchone()
    conn.close()
    
    if path_row and path_row[0]:
        return path_row[0]
    else:
        raise ValueError(f"No file_path found for dataset {dataset_id}")

def get_sessions_for_project(project_name):
    conn = get_db_connection()
    if not conn:
        return []
    cursor = conn.cursor(dictionary=True)
    project_id = get_project_id(project_name)
    query = "SELECT * FROM sessions WHERE project_id = %s"
    cursor.execute(query, (project_id,))
    rows = cursor.fetchall()
    conn.close()
    
    for row in rows:
        if 'bouts' in row and row['bouts']:
            try:
                row['bouts'] = json.loads(row['bouts'])
            except json.JSONDecodeError:
                row['bouts'] = []
    
    return rows

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
            # TODO: zero pad the data to window size
            print(f"Skipping session {session_name} due to insufficient data length: {data.shape[0]} < {window_size}")
            continue

        windowed_data = data.unfold(dimension=0,size=window_size,step=window_stride)
        X.append(windowed_data[:,:-1,:])
        y.append(windowed_data[:,-1,:])

    X = torch.cat(X)
    y = (~(torch.cat(y) == 0).all(axis=1)).float()
    return X,y

fs = 50
window_size = 60 * fs
window_stride = 60 * fs
test_size = 0.2
experiment_name = '2'
labeling = 'andrew smoking labels'

X_train = []
y_train = []
X_test = []
y_test = []

for project_name in ['tonmoy_phase1_imported_20250814_155103','tonmoy_phase2_imported_20250814_155151','tonmoy_phase3_imported_20250814_155204_imported_20250818_161026']:
    print(f"Processing project: {project_name}")
    raw_dataset_path = get_raw_dataset_path(project_name)

    sessions = get_sessions_for_project(project_name)
    sessions = [s for s in sessions if s.get('keep') != 0]

    train_sessions, test_sessions = train_test_split(sessions, test_size=test_size, random_state=42)
    X,y = make_windowed_dataset_from_sessions(train_sessions, window_size, window_stride, raw_dataset_path, labeling)
    X_train.append(X)
    y_train.append(y)

    X,y = make_windowed_dataset_from_sessions(test_sessions, window_size, window_stride, raw_dataset_path, labeling)
    X_test.append(X)
    y_test.append(y)

X_train = torch.cat(X_train)
y_train = torch.cat(y_train)
X_test = torch.cat(X_test)
y_test = torch.cat(y_test)

print(f"X_train samples: {len(X_train)}, y_train samples: {len(y_train)}, X_test samples: {len(X_test)}, y_test samples: {len(y_test)}")
print(f"y_train bincount: {torch.bincount(y_train.long())}, y_test bincount: {torch.bincount(y_test.long())}")
print(f"y_train proportion: {torch.bincount(y_train.long())/len(y_train)}, y_test bincount: {torch.bincount(y_test.long())/len(y_test)}")

os.makedirs(f'experiments/{experiment_name}', exist_ok=True)
torch.save((X_train,y_train),f'experiments/{experiment_name}/train.pt')
torch.save((X_test,y_test),f'experiments/{experiment_name}/test.pt')