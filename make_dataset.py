import os
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

# Load configuration
config = load_config()
experiment_dir = get_experiment_dir(config)

X_train = []
y_train = []
X_test = []
y_test = []

# Track session data and splits for summary
session_data = {}
train_test_splits = {}

for participant in config['participants']:
    print(f"Processing participant: {participant}")
    
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
        sessions = [s for s in sessions if s.get('keep') != 0]

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

        X, y = make_windowed_dataset_from_sessions(
            train_sessions, 
            config['dataset']['window_size'], 
            config['dataset']['window_stride'], 
            raw_dataset_path, 
            config['dataset']['labeling']
        )
        X_train.append(X)
        y_train.append(y)

        X, y = make_windowed_dataset_from_sessions(
            test_sessions, 
            config['dataset']['window_size'], 
            config['dataset']['window_stride'], 
            raw_dataset_path, 
            config['dataset']['labeling']
        )
        X_test.append(X)
        y_test.append(y)

X_train = torch.cat(X_train)
y_train = torch.cat(y_train)
X_test = torch.cat(X_test)
y_test = torch.cat(y_test)

print(f"X_train samples: {len(X_train)}, y_train samples: {len(y_train)}, X_test samples: {len(X_test)}, y_test samples: {len(y_test)}")
print(f"y_train bincount: {torch.bincount(y_train.long())}, y_test bincount: {torch.bincount(y_test.long())}")
print(f"y_train proportion: {torch.bincount(y_train.long())/len(y_train)}, y_test proportion: {torch.bincount(y_test.long())/len(y_test)}")

os.makedirs(experiment_dir, exist_ok=True)
torch.save((X_train, y_train), f'{experiment_dir}/train.pt')
torch.save((X_test, y_test), f'{experiment_dir}/test.pt')

# Generate comprehensive dataset summary
print("\n" + "="*80)
print("GENERATING DATASET SUMMARY")
print("="*80)

summary_results = generate_dataset_summary(
    config=config,
    session_data=session_data,
    train_test_splits=train_test_splits,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    save_dir=experiment_dir
)

print(f"\n✅ Dataset creation and summary complete!")
print(f"   💾 Tensors saved: {experiment_dir}/train.pt, {experiment_dir}/test.pt")
print(f"   📊 Summary files saved in: {experiment_dir}")
print(f"\n📈 Quick Stats:")
print(f"   • Total samples: {len(X_train) + len(X_test):,}")
print(f"   • Train/Test: {len(X_train):,} / {len(X_test):,}")
print(f"   • Class balance: {summary_results['summary_stats']['windowed_data_summary']['overall_positive_percentage']:.1f}% positive")
print(f"   • Projects processed: {len(session_data)}")
print(f"   • Sessions processed: {sum(len(sessions) for sessions in session_data.values())}")
print(f"\n💡 Review the detailed summary files for complete dataset statistics!")
