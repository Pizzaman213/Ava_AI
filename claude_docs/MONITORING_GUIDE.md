# Training Monitoring Guide

## Quick Answer: WandB Status

**Current Status**: ❌ DISABLED

WandB is currently set to `use_wandb: false` in the config file.

## Available Monitoring Options

### 1. Terminal Monitors (Ready to Use!)

#### Option A: Live Log Monitor (Colorized)
```bash
cd /project/code/scripts/5_training
./monitor_training.sh
```
**Features**:
- Real-time log streaming
- Color-coded output:
  - 🟢 Green: Generation tests
  - 🔵 Blue: Loss values
  - 🟡 Yellow: Epoch/batch info
  - 🔴 Red: Errors
  - 🟣 Purple: Validation

#### Option B: Dashboard View (Stats + Log)
```bash
cd /project/code/scripts/5_training
./monitor_dashboard.sh
```
**Features**:
- Auto-refreshing dashboard (every 2 seconds)
- Shows:
  - Process status (PID)
  - Current epoch/batch
  - Latest loss value
  - Generation count
  - GPU memory usage
  - GPU utilization
  - Last 10 log lines

#### Option C: Generation-Only Monitor
```bash
cd /project/code/scripts/5_training
./monitor_generations.sh
```
**Features**:
- Shows ONLY generation outputs
- Filters out training logs
- Easy to read text samples
- Color-coded prompts and outputs

#### Option D: Simple Tail (Basic)
```bash
cd /project/code/scripts/5_training
tail -f nohup.out
```

### 2. Enable WandB (Optional)

If you want to enable Weights & Biases tracking:

#### Step 1: Install WandB
```bash
pip install wandb
```

#### Step 2: Login to WandB
```bash
wandb login
```
(You'll need a WandB account and API key)

#### Step 3: Enable in Config
Edit `/project/code/configs/moe/finetune_from_checkpoint.yaml`:
```yaml
wandb:
  use_wandb: true  # Change from false to true
  project: ava-finetuning
  entity: your-wandb-username  # Add your username
  tags:
    - finetune
    - instruction
    - qa
```

#### Step 4: Restart Training
```bash
cd /project/code/scripts/5_training
bash start_training_with_generation.sh
```

#### WandB Features:
- Web-based dashboard
- Interactive loss plots
- System metrics (GPU, CPU, memory)
- Hyperparameter tracking
- Model comparisons
- Shareable links
- Automatic logging

### 3. TensorBoard (Alternative to WandB)

The training script may also support TensorBoard. Check logs for:
```bash
grep -i tensorboard /project/code/scripts/5_training/nohup.out
```

If available, view with:
```bash
tensorboard --logdir=/project/code/outputs/runs/
```

## Monitoring Commands Reference

### Check Training Status
```bash
# Is training running?
ps aux | grep train_100m_full.py

# Get process ID
pgrep -f train_100m_full.py

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Auto-refresh every second
```

### View Specific Outputs
```bash
# Only generation samples
grep -A 5 "Generated text" nohup.out

# Only loss values
grep "Avg Loss" nohup.out

# Only epoch completions
grep "Epoch.*completed" nohup.out

# Last 50 lines
tail -50 nohup.out

# Follow from line 1000
tail -n +1000 nohup.out | head -100
```

### Search Training Log
```bash
# Find errors
grep -i error nohup.out

# Find warnings
grep -i warning nohup.out

# Find checkpoints
grep -i "checkpoint" nohup.out

# Count generations
grep -c "Testing generation" nohup.out
```

## Recommended Monitoring Setup

### For Active Monitoring (You're watching):
```bash
./monitor_dashboard.sh
```
- Shows everything at a glance
- Updates automatically
- Easy to spot issues

### For Generation Quality Tracking:
```bash
./monitor_generations.sh
```
- See how text quality improves
- Track coherence over time
- No clutter from training logs

### For Detailed Debugging:
```bash
./monitor_training.sh
```
- Full log output
- Color-coded for readability
- See everything that's happening

### For Background Monitoring (Check periodically):
Just check the log file:
```bash
tail -100 nohup.out
```

## Quick Monitoring Workflow

1. **Start Training**:
   ```bash
   cd /project/code/scripts/5_training
   bash start_training_with_generation.sh
   ```

2. **Open Monitor** (in same or new terminal):
   ```bash
   ./monitor_dashboard.sh
   ```
   OR
   ```bash
   ./monitor_generations.sh
   ```

3. **Check GPU** (in another terminal if needed):
   ```bash
   watch -n 1 nvidia-smi
   ```

## Current Training Session

- **Status**: Running ✓
- **Config**: finetune_from_checkpoint.yaml
- **Log File**: /project/code/scripts/5_training/nohup.out
- **Generation Testing**: Every 10 steps
- **WandB**: Disabled (can be enabled)

## Monitor Scripts Created

All located in: `/project/code/scripts/5_training/`

1. ✅ `monitor_training.sh` - Colorized live log
2. ✅ `monitor_dashboard.sh` - Dashboard with stats
3. ✅ `monitor_generations.sh` - Generation outputs only
4. ✅ `start_training_with_generation.sh` - Start script

All scripts are executable and ready to use!

---
**Note**: Terminal monitors don't require any installation or login. 
WandB requires account + API key but provides web-based visualization.
