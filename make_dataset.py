projects = ['unk1','Tonmoy Phase 3']
labeling = f'andrew smoking labels'
window_size = 60
window_stride = 60
test_size = 0.2
experiment_name = '1'

from utils import *

X_train = []
y_train = []
X_test = []
y_test = []

for project_name in projects:
    print(f"Processing project: {project_name}")
    data = get_verified_and_not_deleted_sessions(project_name, labeling)
    project_path = get_project_path(project_name)
    sessions = data['sessions']

    if len(sessions) == 0:
        print(f"No verified sessions found for project {project_name} with labeling {labeling}.")
        continue

    train_sessions, test_sessions = train_test_split(data['sessions'], test_size=test_size, random_state=42)

    print(f"Train sessions size: {len(train_sessions)}, Test sessions size: {len(test_sessions)}")
    # Get total duration and bouts for train and test sessions
    total_duration_train = sum(session['session_duration'] for session in train_sessions)
    total_bouts_train = sum(session['bout_duration'] for session in train_sessions)
    total_duration_test = sum(session['session_duration'] for session in test_sessions)
    total_bouts_test = sum(session['bout_duration'] for session in test_sessions)
    print(f"Total duration for train sessions: {total_duration_train:.2f} seconds, Total bouts: {total_bouts_train}")
    print(f"Total duration for test sessions: {total_duration_test:.2f} seconds, Total bouts: {total_bouts_test}")

    X,y = make_windowed_dataset_from_sessions(train_sessions, window_size, window_stride, project_path)
    X_train.append(X)
    y_train.append(y)

    X,y = make_windowed_dataset_from_sessions(test_sessions, window_size, window_stride, project_path)
    X_test.append(X)
    y_test.append(y)

# print number of samples and bincount for each of X_train, y_train, X_test, y_test

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