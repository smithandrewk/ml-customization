# Distributed Training Guide

This guide explains how to run training jobs across multiple servers and GPUs using `run_distributed_training.py`.

## Setup

1. **Configure SSH keys** (first time only):
   ```bash
   # Generate SSH key if you don't have one
   ssh-keygen -t rsa -b 4096

   # Copy to each server
   ssh-copy-id andrew@10.173.98.203
   ssh-copy-id andrew@10.173.98.204
   ```

2. **Edit cluster configuration** (`cluster_config.json`):
   ```json
   {
     "servers": [
       {
         "host": "10.173.98.203",
         "user": "andrew",
         "gpus": 2,
         "ssh_key": "~/.ssh/id_rsa"
       },
       {
         "host": "10.173.98.204",
         "user": "andrew",
         "gpus": 2,
         "ssh_key": "~/.ssh/id_rsa"
       }
     ],
     "script_path": "train.py"
   }
   ```

3. **Create job configuration** (`jobs_config.json`):
   ```json
   [
     {"fold": 0, "batch_size": 64, "model": "medium", "prefix": "exp1"},
     {"fold": 1, "batch_size": 64, "model": "medium", "prefix": "exp1"},
     {"fold": 2, "batch_size": 64, "model": "medium", "prefix": "exp1"}
   ]
   ```

## Running Jobs

```bash
python run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --log-file results.json
```

The script will:
- Create a tmux session named `ml_training` on each server
- Split the session into one pane per GPU
- Distribute jobs across all available GPUs
- Run jobs in parallel

## Monitoring Progress

### View training output on a server:

```bash
# Connect to server and attach to tmux session
ssh andrew@10.173.98.203 -t 'tmux attach -t ml_training'

# Navigate between panes:
# Ctrl+b, arrow keys - move between panes
# Ctrl+b, z - zoom in/out of current pane
# Ctrl+b, d - detach (jobs keep running)
```

### Tmux quick reference:

- `Ctrl+b` then `↑/↓/←/→` - Navigate between panes
- `Ctrl+b` then `z` - Toggle pane zoom (fullscreen current pane)
- `Ctrl+b` then `d` - Detach from session (jobs continue running)
- `Ctrl+b` then `[` - Enter scroll mode (use arrow keys, `q` to exit)

## Output

- **Console**: Real-time progress updates
- **Log file**: JSON file with detailed results (if `--log-file` specified)
- **Experiments**: Each job creates output in `experiments/{prefix}/fold{N}_{participant}/`

## Example Workflow

```bash
# 1. Start training
python run_distributed_training.py \
    --cluster-config cluster_config.json \
    --jobs-config jobs_config.json \
    --log-file results.json

# 2. Monitor a specific server (in another terminal)
ssh andrew@10.173.98.203 -t 'tmux attach -t ml_training'

# 3. After completion, analyze results
cat results.json
```

## Tips

- Jobs run in parallel across all GPUs automatically
- The `device` parameter is set automatically - don't specify it in job configs
- Tmux sessions persist even if you disconnect
- Each pane shows output for one GPU
- All paths are relative to `~/ml-customization` on remote servers
