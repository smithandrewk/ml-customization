# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project focused on smoking detection from accelerometer data. The project uses PyTorch to train CNN models for binary classification of smoking behavior based on time-series accelerometer data from wearable devices.

## Key Components

### Core Modules
- **`utils.py`**: Central utility module containing:
  - Database connection and query functions for MySQL backend
  - Data loading and preprocessing utilities
  - `SmokingCNN` model architecture (1D CNN with dilated convolutions)
  - Training visualization functions (`plot_training_progress`)
  - Dataset creation functions (`make_windowed_dataset_from_sessions`)

### Training Scripts
- **`train.py`**: Main training script that loads preprocessed datasets and trains the SmokingCNN model with early stopping based on F1 score
- **`make_dataset_participants.py`**: Creates train/test datasets by splitting sessions across multiple participant projects
- **`make_dataset_v2.py`**: Alternative dataset creation script with similar functionality

### Data Flow
1. Raw accelerometer data is stored in MySQL database with session metadata
2. Dataset creation scripts query the database and load CSV files from the filesystem
3. Data is windowed (60-second windows at 50Hz sampling rate = 3000 samples per window)
4. Labels are applied based on smoking bout annotations stored in the database
5. Models are trained with binary cross-entropy loss and early stopping

## Development Commands

### Environment Setup
```bash
# Ensure Python 3.10+ is available
python3 --version

# Install dependencies (no requirements.txt found - dependencies are imported directly)
# Key dependencies: torch, mysql-connector-python, pandas, matplotlib, sklearn, numpy, python-dotenv, pyyaml
```

### Configuration Management
The project uses YAML configuration files for managing training parameters:

```bash
# Use default configuration
python3 make_dataset_participants.py
python3 train.py

# Use custom configuration
python3 make_dataset_participants.py --config experiments/experiment_3.yaml
python3 train.py --config experiments/experiment_3.yaml
```

### Dataset Creation
```bash
# Create dataset from multiple participant projects (generates comprehensive summary)
python3 make_dataset_participants.py

# Alternative dataset creation approach
python3 make_dataset_v2.py
```

The dataset creation process automatically generates detailed summaries including:
- **Session-level CSV** (`dataset_sessions_TIMESTAMP.csv`): Individual session statistics, durations, smoking bouts, train/test assignments
- **Project-level CSV** (`dataset_projects_TIMESTAMP.csv`): Aggregated statistics per project
- **Comprehensive text report** (`dataset_summary_TIMESTAMP.txt`): Full dataset overview with class distributions, split statistics, and configuration details

Output files are timestamped and saved in the experiment directory for reproducibility.

### Model Training
```bash
# Train smoking detection model
python3 train.py
```

## Architecture Notes

### Database Schema
- Uses MySQL database (`delta2`) with tables for `projects`, `sessions`, `participants`, `raw_datasets`, and `project_dataset_refs`
- Session data includes `bouts` field with JSON-encoded smoking annotations
- Raw accelerometer data stored as CSV files referenced by database paths

### Model Architecture
- 1D CNN (`SmokingCNN`) designed for 3000-sample windows (60 seconds @ 50Hz)
- Uses dilated convolutions to capture long-range temporal dependencies
- Global average pooling with dropout for regularization
- Binary classification output with sigmoid activation

### Training Configuration
Configuration is managed through YAML files:
- **Main config**: `config.yaml` - Default training parameters
- **Experiment configs**: `experiments/experiment_X.yaml` - Experiment-specific overrides
- **Key parameters**: Window size, batch size, learning rate, early stopping patience
- **Experiments saved**: `experiments/{experiment_name}/` directory structure

### Environment Variables
The project uses `.env` file for database configuration:
- `MYSQL_HOST`
- `MYSQL_USER` 
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

Note: Some files contain hardcoded database credentials which should be migrated to environment variables for security.