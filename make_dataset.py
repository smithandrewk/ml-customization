import os
import argparse
import torch
from sklearn.model_selection import train_test_split
from utils import (
    load_config,
    get_experiment_dir,
    make_windowed_dataset_from_sessions,
    get_participant_id,
    get_participant_projects,
    get_raw_dataset_path,
    get_sessions_for_project,
    generate_dataset_summary
)

def get_next_dataset_dir(name):
    """Get next auto-incrementing dataset directory."""
    os.makedirs('data', exist_ok=True)
    
    # Find highest existing number
    existing = [d for d in os.listdir('data') if os.path.isdir(f'data/{d}')]
    numbers = []
    for d in existing:
        if '_' in d:
            try:
                num = int(d.split('_')[0])
                numbers.append(num)
            except ValueError:
                continue
    
    next_num = max(numbers) + 1 if numbers else 1
    dataset_dir = f'data/{next_num:03d}_{name}'
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create participant-specific smoking detection dataset')
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
parser.add_argument('--name', type=str, required=True, help='Dataset name (will be prefixed with auto-incrementing number)')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
experiment_dir = get_next_dataset_dir(args.name)

# Track session data and splits for summary (across all participants)
session_data = {}
train_test_splits = {}
all_participant_data = {}

for participant in config['participants']:
    print(f"Processing participant: {participant}")
    
    # Initialize participant-specific lists
    participant_X_train = []
    participant_y_train = []
    participant_X_test = []
    participant_y_test = []
    
    # Get participant_id from participant code
    participant_id = get_participant_id(participant)
    print(f"Participant ID: {participant_id}")
    
    # Get projects for the participant
    projects = get_participant_projects(participant_id)
    print(f"Projects for {participant}: {projects}")
    
    for project_name in projects:
        print(f"Processing project: {project_name}")
        raw_dataset_path = get_raw_dataset_path(project_name)

        sessions = get_sessions_for_project(project_name)
        sessions = [s for s in sessions if s.get('keep') != 0 and s.get('smoking_verified') == 1]

        if len (sessions) == 0:
            print(f"  No valid sessions found for project {project_name}, skipping.")
            continue
        
        # Store session data for summary
        session_data[project_name] = sessions

        train_sessions, test_sessions = train_test_split(
            sessions, 
            test_size=config['dataset']['test_size'], 
            random_state=42
        )
        
        # Store train/test splits for summary
        train_test_splits[project_name] = (train_sessions, test_sessions)
        
        print(f"Train sessions: {len(train_sessions)}, Test sessions: {len(test_sessions)}")

        # Create windowed datasets for training data
        X, y = make_windowed_dataset_from_sessions(
            train_sessions, 
            config['dataset']['window_size'], 
            config['dataset']['window_stride'], 
            raw_dataset_path, 
            config['dataset']['labeling'],
            config.get('sensors')
        )
        participant_X_train.append(X)
        participant_y_train.append(y)

        # Create windowed datasets for test data
        X, y = make_windowed_dataset_from_sessions(
            test_sessions, 
            config['dataset']['window_size'], 
            config['dataset']['window_stride'], 
            raw_dataset_path, 
            config['dataset']['labeling'],
            config.get('sensors')
        )
        participant_X_test.append(X)
        participant_y_test.append(y)

    # Concatenate all projects for this participant
    if participant_X_train:  # Check if participant has any data
        participant_X_train = torch.cat(participant_X_train)
        participant_y_train = torch.cat(participant_y_train)
        participant_X_test = torch.cat(participant_X_test)
        participant_y_test = torch.cat(participant_y_test)

        # Save participant-specific files
        torch.save((participant_X_train, participant_y_train), f'{experiment_dir}/{participant}_train.pt')
        torch.save((participant_X_test, participant_y_test), f'{experiment_dir}/{participant}_test.pt')

        # Store for summary
        all_participant_data[participant] = {
            'train_samples': len(participant_X_train),
            'test_samples': len(participant_X_test),
            'train_positive': torch.bincount(participant_y_train.long()),
            'test_positive': torch.bincount(participant_y_test.long())
        }

        print(f"Participant {participant}:")
        print(f"  Train samples: {len(participant_X_train):,}")
        print(f"  Test samples: {len(participant_X_test):,}")
        print(f"  Train class distribution: {torch.bincount(participant_y_train.long())}")
        print(f"  Test class distribution: {torch.bincount(participant_y_test.long())}")
        print(f"  Files saved: {participant}_train.pt, {participant}_test.pt")
        print()

print(f"\n✅ Participant-specific dataset creation complete!")
print(f"   💾 Files saved in: {experiment_dir}")
print(f"\n📊 Summary by participant:")

total_train_samples = 0
total_test_samples = 0

for participant, data in all_participant_data.items():
    train_pos = data['train_positive'][1] if len(data['train_positive']) > 1 else 0
    test_pos = data['test_positive'][1] if len(data['test_positive']) > 1 else 0
    train_total = data['train_samples']
    test_total = data['test_samples']
    
    total_train_samples += train_total
    total_test_samples += test_total
    
    print(f"  • {participant}: {train_total:,} train, {test_total:,} test")
    print(f"    - Train: {train_pos}/{train_total} ({train_pos/train_total*100:.1f}% positive)")
    print(f"    - Test: {test_pos}/{test_total} ({test_pos/test_total*100:.1f}% positive)")

print(f"\n💡 Total across all participants:")
print(f"   • Training samples: {total_train_samples:,}")
print(f"   • Test samples: {total_test_samples:,}")
print(f"\n🔄 Usage examples:")
print(f"   # Train on single participant")
print(f"   python3 train.py --dataset_dir {experiment_dir} --participant tonmoy")
print(f"   ")
print(f"   # Train on all participants (concatenated)")
print(f"   python3 train.py --dataset_dir {experiment_dir}")