# Dataset Creation Process Documentation

## Overview

The `make_dataset.py` script is responsible for creating participant-specific smoking detection datasets from raw accelerometer and gyroscope sensor data. It processes time-series data from wearable devices, applies windowing transformations, and generates train/validation/test splits for machine learning model training.

## Purpose

Transform raw sensor data from participant sessions into standardized, windowed datasets suitable for training CNN-based smoking detection models. Each participant receives their own set of training, validation, and test files to enable both participant-specific and cross-participant model training.

## Input Requirements

### 1. Configuration File (`configs/dataset_config.yaml`)

**Dataset Parameters:**
- `fs`: **50 Hz** - Sampling frequency of the sensor data
- `window_size_seconds`: **60 seconds** - Duration of each data window
- `window_stride_seconds`: **60 seconds** - Step size between windows (non-overlapping)
- `labeling`: **"andrew smoking labels"** - Label type identifier for database queries
- `test_size`: **0.2** - Default train/test split ratio (currently unused; all participants use 60/20/20 split)

**Sensor Configuration:**
- `use_accelerometer`: **true** - Include accelerometer data (accel_x, accel_y, accel_z)
- `use_gyroscope`: **true** - Include gyroscope data (gyro_x, gyro_y, gyro_z)

**Participants:**
- List of participant codes (e.g., `["ashlin"]`)

### 2. Command Line Arguments

```bash
python3 make_dataset.py --config <path_to_config> --name <dataset_name>
```

- `--config`: Path to configuration YAML file (default: `configs/dataset_config.yaml`)
- `--name`: Dataset name for output directory (required)

### 3. Data Sources

**Database:**
- MySQL database containing:
  - Participant metadata (participant IDs, codes)
  - Project information
  - Session metadata (with `keep` and `smoking_verified` flags)
  - Smoking bout annotations (JSON-encoded in `bouts` field)

**Filesystem:**
- Raw CSV files containing time-series sensor data
- Paths retrieved from database via `get_raw_dataset_path()`

## Processing Pipeline

### Step 1: Output Directory Creation

Creates auto-incrementing dataset directories:
```
data/001_dataset_name/
data/002_another_dataset/
...
```

The script finds the highest existing number and increments to avoid overwriting.

### Step 2: Participant Processing Loop

For each participant in the configuration:

1. **Participant Identification**
   - Resolves participant code → participant ID via `get_participant_id()`

2. **Project Retrieval**
   - Gets all projects associated with the participant
   - Multiple projects per participant are supported

3. **Session Filtering**
   - Retrieves all sessions for each project
   - Filters sessions with:
     - `keep != 0` (sessions marked for retention)
     - `smoking_verified == 1` (sessions with verified smoking annotations)

4. **Data Splitting** (60/20/20 Split)

   All participants receive a 3-way split:
   - **60% Training**: For model training
   - **20% Validation**: For hyperparameter tuning and model selection
   - **20% Test**: For final evaluation

   ```python
   # First split: 60% train, 40% temp
   train_sessions, temp_sessions = train_test_split(sessions, test_size=0.4, random_state=42)

   # Second split: 20% val, 20% test (from the 40% temp)
   val_sessions, test_sessions = train_test_split(temp_sessions, test_size=0.5, random_state=42)
   ```

5. **Windowing Transformation**

   Each session's continuous sensor data is converted into fixed-size windows:
   - **Window size**: 60 seconds × 50 Hz = **3000 samples per window**
   - **Window stride**: 60 seconds (non-overlapping windows)
   - **Sensors**: 6 channels (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
   - **Window shape**: `[3000, 6]` per window

   Applied separately to:
   - Training sessions → `participant_X_train`, `participant_y_train`
   - Validation sessions → `participant_X_val`, `participant_y_val`
   - Test sessions → `participant_X_test`, `participant_y_test`

6. **Label Assignment**

   Labels are applied using the specified labeling strategy (`"andrew smoking labels"`):
   - Binary classification: **0** (not smoking) or **1** (smoking)
   - Labels determined by smoking bout annotations in the database
   - Each window receives a single label based on its temporal overlap with smoking bouts

7. **Data Aggregation**

   All projects for a participant are concatenated:
   ```python
   participant_X_train = torch.cat([project1_X_train, project2_X_train, ...])
   participant_y_train = torch.cat([project1_y_train, project2_y_train, ...])
   ```

### Step 3: File Serialization

Saves PyTorch tensor files for each participant:

```
data/001_dataset_name/
├── ashlin_train.pt    # (X_train, y_train) tuple
├── ashlin_val.pt      # (X_val, y_val) tuple
├── ashlin_test.pt     # (X_test, y_test) tuple
├── participant2_train.pt
├── participant2_val.pt
└── participant2_test.pt
```

Each `.pt` file contains a tuple: `(X, y)`
- `X`: Tensor of shape `[num_windows, 3000, 6]` (windows × samples × channels)
- `y`: Tensor of shape `[num_windows]` (binary labels)

## Output Data Structure

### Per-Participant Files

Each participant gets 3 files:

1. **`{participant}_train.pt`**
   - Training data for model optimization
   - Typically ~60% of total data

2. **`{participant}_val.pt`**
   - Validation data for hyperparameter tuning
   - Typically ~20% of total data
   - Used during training for early stopping

3. **`{participant}_test.pt`**
   - Test data for final evaluation
   - Typically ~20% of total data
   - Never used during training

### Summary Statistics

The script prints comprehensive statistics:

**Per-Participant Summary:**
```
Participant ashlin:
  Train samples: 1,234
  Val samples: 412
  Test samples: 411
  Train class distribution: tensor([987, 247])  # [negative, positive]
  Val class distribution: tensor([329, 83])
  Test class distribution: tensor([328, 83])
  Files saved: ashlin_train.pt, ashlin_val.pt, ashlin_test.pt
```

**Global Summary:**
```
Total across all participants:
   • Training samples: 1,234
   • Test samples: 411
```

**Usage Examples:**
```bash
# Train on single participant
python3 train.py --dataset_dir data/001_dataset_name --participant ashlin

# Train on all participants (concatenated)
python3 train.py --dataset_dir data/001_dataset_name
```

## Data Characteristics

### Window Properties
- **Sampling rate**: 50 Hz
- **Window duration**: 60 seconds
- **Samples per window**: 3,000
- **Overlap**: None (stride = window size)
- **Sensor channels**: 6 (3 accelerometer + 3 gyroscope)

### Tensor Dimensions
- **Input shape**: `[batch_size, 3000, 6]`
- **Label shape**: `[batch_size]`
- **Data type**: PyTorch tensors (float32 for X, long for y)

### Class Distribution
- **Class 0**: Not smoking (typically majority class)
- **Class 1**: Smoking (typically minority class)
- Class imbalance is common and expected

## Session Filtering Logic

Only sessions meeting ALL criteria are included:
1. `keep != 0`: Session is marked for retention (not excluded for quality reasons)
2. `smoking_verified == 1`: Smoking annotations have been verified by human annotators

Sessions failing these criteria are skipped with a warning message.

## Split Reproducibility

- All splits use `random_state=42` for reproducibility
- Same configuration will always produce identical train/val/test splits
- Splits are performed at the **session level** (not window level)
  - Ensures all windows from a session stay together
  - Prevents data leakage between splits

## Usage Workflow

### Creating a New Dataset

```bash
# 1. Configure participants and parameters
nano configs/dataset_config.yaml

# 2. Run dataset creation
python3 make_dataset.py --config configs/dataset_config.yaml --name experiment_1

# 3. Output appears in data/001_experiment_1/

# 4. Train a model using the dataset
python3 train.py --dataset_dir data/001_experiment_1 --participant ashlin
```

### Multi-Participant Datasets

To create datasets for multiple participants, list them in the config:

```yaml
participants:
  - "ashlin"
  - "tonmoy"
  - "participant_3"
```

Each participant will have their own train/val/test files, enabling:
- **Participant-specific models**: Train on one participant's data
- **Cross-participant models**: Concatenate all participants' training data
- **Transfer learning**: Pre-train on multiple participants, fine-tune on target participant

## Key Functions

- `get_participant_id(participant_code)`: Maps participant code → database ID
- `get_participant_projects(participant_id)`: Retrieves all projects for a participant
- `get_raw_dataset_path(project_name)`: Gets filesystem path to raw CSV data
- `get_sessions_for_project(project_name)`: Retrieves session metadata from database
- `make_windowed_dataset_from_sessions()`: Core windowing logic
  - Loads raw CSV data
  - Applies sliding window
  - Assigns labels based on smoking bouts
  - Returns PyTorch tensors

## Error Handling

- **No valid sessions**: Project is skipped with warning message
- **Missing participant data**: Participant is skipped if no data available
- **Database connection failures**: Script will crash (handled by database utilities)

## Notes

- The current implementation treats ALL participants as "target participants" (line 84: `is_target_participant = True`)
- The legacy 2-way split code (lines 102-110) is currently unreachable but preserved for reference
- Dataset directories are auto-numbered to prevent accidental overwrites
- All validation data handling uses conditional checks (`if 'participant_X_val' in locals()`) to support both 2-way and 3-way splits
