import os
import argparse
import torch
from sklearn.model_selection import train_test_split
from lib.utils import (
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
parser.add_argument('--config', type=str, default='configs/dataset_config.yaml', help='Path to config file')
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

        # Check splitting strategy
        split_by_windows = config['dataset'].get('split_by_windows', False)

        if split_by_windows:
            # NEW APPROACH: Create all windows first, then split at window level
            print(f"Using window-level splitting (temporal leakage possible)")

            X_all, y_all = make_windowed_dataset_from_sessions(
                sessions,  # All sessions together
                config['dataset']['window_size'],
                config['dataset']['window_stride'],
                raw_dataset_path,
                config['dataset']['labeling'],
                config.get('sensors')
            )

            # 3-way split: 60% train, 20% val, 20% test
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_all, y_all,
                test_size=0.4,  # 40% for val+test
                random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.5,  # 50% of 40% = 20% total for test
                random_state=42
            )

            # Append to participant lists
            participant_X_train.append(X_train)
            participant_y_train.append(y_train)

            if 'participant_X_val' not in locals():
                participant_X_val = []
                participant_y_val = []
            participant_X_val.append(X_val)
            participant_y_val.append(y_val)

            participant_X_test.append(X_test)
            participant_y_test.append(y_test)

            print(f"Window-level split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        else:
            # CURRENT APPROACH: Split sessions first, then create windows (prevents temporal leakage)
            print(f"Using session-level splitting (prevents temporal leakage)")

            # Generate train/val/test splits for all participants
            # train.py will decide how to use them based on target vs base
            is_target_participant = True  # All participants get 3-way split

            if is_target_participant:
                # 3-way split for target participants: 60% train, 20% val, 20% test
                train_sessions, temp_sessions = train_test_split(
                    sessions,
                    test_size=0.4,  # 40% for val+test
                    random_state=42
                )
                val_sessions, test_sessions = train_test_split(
                    temp_sessions,
                    test_size=0.5,  # 50% of 40% = 20% total for test
                    random_state=42
                )
                # Store 3-way split for target participants
                train_test_splits[project_name] = (train_sessions, val_sessions, test_sessions)
                print(f"Target participant split - Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
            else:
                # 2-way split for base participants: train/test (test serves as validation)
                train_sessions, test_sessions = train_test_split(
                    sessions,
                    test_size=config['dataset']['test_size'],
                    random_state=42
                )
                # Store 2-way split for base participants (val = test)
                train_test_splits[project_name] = (train_sessions, test_sessions, test_sessions)
                print(f"Base participant split - Train: {len(train_sessions)}, Test/Val: {len(test_sessions)}")

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

            # Create windowed datasets for validation data (only for target participants)
            if is_target_participant and len(train_test_splits[project_name]) == 3:
                val_sessions = train_test_splits[project_name][1]  # Get validation sessions
                X_val, y_val = make_windowed_dataset_from_sessions(
                    val_sessions,
                    config['dataset']['window_size'],
                    config['dataset']['window_stride'],
                    raw_dataset_path,
                    config['dataset']['labeling'],
                    config.get('sensors')
                )
                if 'participant_X_val' not in locals():
                    participant_X_val = []
                    participant_y_val = []
                participant_X_val.append(X_val)
                participant_y_val.append(y_val)

            # Create windowed datasets for test data
            test_sessions_for_data = train_test_splits[project_name][-1]  # Last element is always test
            X, y = make_windowed_dataset_from_sessions(
                test_sessions_for_data,
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

        # Handle validation data for target participants
        has_validation = 'participant_X_val' in locals() and participant_X_val
        if has_validation:
            participant_X_val = torch.cat(participant_X_val)
            participant_y_val = torch.cat(participant_y_val)

        # Save participant-specific files
        torch.save((participant_X_train, participant_y_train), f'{experiment_dir}/{participant}_train.pt')
        torch.save((participant_X_test, participant_y_test), f'{experiment_dir}/{participant}_test.pt')
        
        files_saved = [f'{participant}_train.pt', f'{participant}_test.pt']
        if has_validation:
            torch.save((participant_X_val, participant_y_val), f'{experiment_dir}/{participant}_val.pt')
            files_saved.append(f'{participant}_val.pt')

        # Store for summary
        summary_data = {
            'train_samples': len(participant_X_train),
            'test_samples': len(participant_X_test),
            'train_positive': torch.bincount(participant_y_train.long()),
            'test_positive': torch.bincount(participant_y_test.long())
        }
        
        if has_validation:
            summary_data.update({
                'val_samples': len(participant_X_val),
                'val_positive': torch.bincount(participant_y_val.long())
            })
        
        all_participant_data[participant] = summary_data

        print(f"Participant {participant}:")
        print(f"  Train samples: {len(participant_X_train):,}")
        if has_validation:
            print(f"  Val samples: {len(participant_X_val):,}")
        print(f"  Test samples: {len(participant_X_test):,}")
        print(f"  Train class distribution: {torch.bincount(participant_y_train.long())}")
        if has_validation:
            print(f"  Val class distribution: {torch.bincount(participant_y_val.long())}")
        print(f"  Test class distribution: {torch.bincount(participant_y_test.long())}")
        print(f"  Files saved: {', '.join(files_saved)}")
        print()
        
        # Reset validation variables for next participant
        if 'participant_X_val' in locals():
            del participant_X_val, participant_y_val

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