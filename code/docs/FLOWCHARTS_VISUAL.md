# Ava LLM Training Framework - Visual Flowcharts 

Beautiful, color-coded flowcharts for the Ava training system.

---

## Training Lifecycle 

### Complete Training Flow

```mermaid
graph TD
    A[ Start Training] --> B[ Load Configuration]
    B --> C[ Initialize Data Pipeline]
    C --> D[ Initialize Model]
    D --> E[ Initialize Optimizer]
    E --> F[ Warmup Phase]
    F --> G[ Training Loop]

    G --> H[ Get Batch]
    H --> I[ Forward Pass]
    I --> J[ Compute Loss]
    J --> K[ Backward Pass]
    K --> L[ Clip Gradients]
    L --> M[ Optimizer Step]
    M --> N[ Update Learning Rate]

    N --> O{ Eval Step?}
    O -->|Yes| P[ Run Evaluation]
    O -->|No| Q{ Save Step?}
    P --> Q

    Q -->|Yes| R[ Save Checkpoint]
    Q -->|No| S{ More Steps?}
    R --> S

    S -->|Yes| G
    S -->|No| T[ Final Evaluation]
    T --> U[ Save Final Model]
    U --> V[ Training Complete]

    style A fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style V fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style G fill:#2196F3,stroke:#1565C0,stroke-width:3px,color:#fff
    style I fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style K fill:#FF9800,stroke:#E65100,stroke-width:2px,color:#fff
    style P fill:#9C27B0,stroke:#6A1B9A,stroke-width:2px,color:#fff
    style R fill:#00BCD4,stroke:#006064,stroke-width:2px,color:#fff
```

### Single Training Step Detail

```mermaid
graph LR
    A[ Input Batch] --> B[ Forward Pass]
    B --> C[ Loss Computation]
    C --> D[ Backward Pass]
    D --> E[ Gradients]
    E --> F[ Gradient Clipping]
    F --> G[ Optimization]
    G --> H[ Parameter Update]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style E fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style F fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    style G fill:#E0F2F1,stroke:#009688,stroke-width:2px
    style H fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
```

---

## Model Architecture 

### Transformer Layer Flow

```mermaid
graph TD
    A[ Input: B×L×H] --> B[ Layer Norm 1]
    B --> C[ Multi-Head Attention]
    C --> D[ Residual Connection]
    D --> E[ Layer Norm 2]
    E --> F[ MoE Router]

    F --> G[ Expert 1]
    F --> H[ Expert 2]
    F --> I[ Expert 3]
    F --> J[ Expert 8]

    G --> K[ Weighted Combine]
    H --> K
    I --> K
    J --> K

    K --> L[ Residual Connection]
    L --> M[ Output: B×L×H]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style C fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style F fill:#FCE4EC,stroke:#E91E63,stroke-width:3px
    style G fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style H fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style I fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style J fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style K fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
    style M fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
```

### Attention Mechanism Flow

```mermaid
graph LR
    A[ Hidden States] --> B[ Split Q, K, V]
    B --> C[ Apply RoPE]
    C --> D[ Q @ K^T]
    D --> E[ Scale by √d]
    E --> F[ Apply Mask]
    F --> G[ Softmax]
    G --> H[ @ Values]
    H --> I[ Concat Heads]
    I --> J[ Output Projection]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style G fill:#FFE082,stroke:#F57C00,stroke-width:3px
    style J fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
```

---

## Expert Routing 

### MoE Router Decision Flow

```mermaid
graph TD
    A[ Token Hidden State] --> B[ Router Linear Layer]
    B --> C[ Add Jitter Noise]
    C --> D[ Softmax Normalization]
    D --> E[ Top-K Selection K=2]

    E --> F[ Expert 1: p₁]
    E --> G[ Expert 2: p₂]

    F --> H{ Capacity OK?}
    H -->| Yes| I[ Route to Expert 1]
    H -->| Full| J[ Overflow Handler]

    G --> K{ Capacity OK?}
    K -->| Yes| L[ Route to Expert 2]
    K -->| Full| M[ Overflow Handler]

    J --> N[ Try Next Best]
    M --> O[ Try Next Best]

    I --> P[ Weighted Sum]
    L --> P
    N --> P
    O --> P

    P --> Q[ Final Output]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style D fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style E fill:#FCE4EC,stroke:#E91E63,stroke-width:3px
    style I fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style L fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style J fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style M fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style P fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
    style Q fill:#80DEEA,stroke:#00838F,stroke-width:3px
```

### Load Balancing Flow

```mermaid
graph LR
    A[ Router Probs] --> B[ Compute Expert Fractions]
    B --> C[ Compute Avg Probs]
    C --> D[ Balance Loss]
    D --> E[ Add to Total Loss]
    E --> F[ Backprop]
    F --> G[ Update Router Weights]
    G --> H[ Balanced Routing]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:3px
    style F fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style G fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style H fill:#80DEEA,stroke:#00838F,stroke-width:3px
```

---

## Gradient Flow 

### Backward Pass and Update

```mermaid
graph TD
    A[ Loss Computed] --> B{ Loss Valid?}
    B -->| NaN/Inf| C[⏭ Skip Step]
    B -->| Valid| D[ Backward Pass]

    D --> E[ Compute Gradients]
    E --> F{ Gradients Valid?}
    F -->| NaN/Inf| C
    F -->| Valid| G[ Compute Grad Norm]

    G --> H{ Norm > Threshold?}
    H -->| Explosion| I[ Reduce LR]
    H -->| Healthy| J[ Clip Gradients]
    I --> J

    J --> K{ Distributed?}
    K -->|Yes| L[ All-Reduce Gradients]
    K -->|No| M[ Optimizer Step]
    L --> M

    M --> N[ Update Parameters]
    N --> O[ Scheduler Step]
    O --> P[ Update LR]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style C fill:#FFCDD2,stroke:#C62828,stroke-width:2px,color:#000
    style D fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    style E fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style I fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style J fill:#FFF9C4,stroke:#F57C00,stroke-width:2px
    style L fill:#E0F2F1,stroke:#00897B,stroke-width:2px
    style M fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
    style N fill:#80DEEA,stroke:#00838F,stroke-width:3px
```

### Gradient Health Monitor

```mermaid
graph LR
    A[ Gradients] --> B[ Compute Norm]
    B --> C{ Health Check}
    C -->| Explosion| D[ Counter++]
    C -->| Healthy| E[ Reset Counter]
    D --> F{ Counter > Window?}
    F -->| Yes| G[ Emergency LR Reduce]
    F -->|No| H[ Continue]
    E --> H
    G --> H

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style C fill:#FFF3E0,stroke:#FF9800,stroke-width:3px
    style D fill:#FFCCBC,stroke:#D84315,stroke-width:2px
    style E fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style G fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style H fill:#80DEEA,stroke:#00838F,stroke-width:2px
```

---

## Memory Management 

### GPU Memory Monitoring

```mermaid
graph TD
    A[ Check GPU Memory] --> B[ Compute Utilization]
    B --> C{ Utilization Level?}

    C -->|< 75%| D[ Normal: Continue]
    C -->|75-80%| E[ Warning: Log Alert]
    C -->|80-85%| F[ Critical: Prepare Action]
    C -->|85-90%| G[ Emergency: Clear Cache]
    C -->|> 90%| H[ Crisis: Reduce Batch]

    E --> D
    F --> I[ Reduce Batch 25%]
    I --> D
    G --> J[ torch.cuda.empty_cache]
    J --> K[ Reduce Batch 50%]
    K --> D
    H --> L[ Save Emergency Checkpoint]
    L --> M[ Set Min Batch Size]
    M --> N[ Raise OOM Error]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style D fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
    style E fill:#FFF9C4,stroke:#F57C00,stroke-width:2px
    style F fill:#FFCC80,stroke:#EF6C00,stroke-width:2px
    style G fill:#FFAB91,stroke:#D84315,stroke-width:2px
    style H fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style N fill:#000,stroke:#000,stroke-width:3px,color:#fff
```

### OOM Recovery Strategy

```mermaid
graph LR
    A[ OOM Error] --> B[ Enable Grad Checkpoint]
    B --> C{ Works?}
    C -->|No| D[ Reduce Batch Size]
    C -->|Yes| E[ Continue Training]
    D --> F{ Works?}
    F -->|No| G[ Enable ZeRO-2]
    F -->|Yes| E
    G --> H{ Works?}
    H -->|No| I[ Enable CPU Offload]
    H -->|Yes| E
    I --> J{ Works?}
    J -->|No| K[ Fatal Error]
    J -->|Yes| E

    style A fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style D fill:#FFCC80,stroke:#EF6C00,stroke-width:2px
    style G fill:#CE93D8,stroke:#8E24AA,stroke-width:2px
    style I fill:#90CAF9,stroke:#1565C0,stroke-width:2px
    style E fill:#81C784,stroke:#388E3C,stroke-width:3px
    style K fill:#000,stroke:#000,stroke-width:3px,color:#fff
```

---

## Checkpoint Management 

### Save Checkpoint Flow

```mermaid
graph TD
    A[ Save Trigger] --> B{ Rank 0?}
    B -->|No| C[⏳ Wait at Barrier]
    B -->|Yes| D[ Create Checkpoint Dict]

    D --> E[ Add Model State]
    E --> F[ Add Optimizer State]
    F --> G[ Add Scheduler State]
    G --> H[ Add Metrics]
    H --> I[ Add RNG States]

    I --> J[ Write to Disk]
    J --> K[ Update 'latest' Link]
    K --> L{ Is Best Model?}
    L -->|Yes| M[ Update 'best' Link]
    L -->|No| N[ Sync Barrier]
    M --> N

    C --> N
    N --> O[ Save Complete]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style D fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style J fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style M fill:#FFD54F,stroke:#F57C00,stroke-width:3px
    style O fill:#81C784,stroke:#388E3C,stroke-width:3px
```

### Load Checkpoint Flow

```mermaid
graph TD
    A[ Load Request] --> B{ Path Valid?}
    B -->|No| C[ Search for 'latest']
    B -->|Yes| D[ Read File]
    C --> D

    D --> E{ Valid Checkpoint?}
    E -->|No| F[ Load Error]
    E -->|Yes| G[ Load Model State]

    G --> H{ Success?}
    H -->|No| F
    H -->|Yes| I[ Load Optimizer State]

    I --> J[ Load Scheduler State]
    J --> K[ Restore RNG States]
    K --> L[ Extract Metrics]
    L --> M[ Load Complete]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style F fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style G fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style I fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style M fill:#81C784,stroke:#388E3C,stroke-width:3px
```

---

## Data Pipeline 

### Data Processing Flow

```mermaid
graph LR
    A[ Raw Files] --> B[ Format Detection]
    B --> C[ Load Data]
    C --> D[ Validation]
    D --> E[ Quality Filters]
    E --> F[ Tokenization]
    F --> G[ Batching]
    G --> H[ DataLoader]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style E fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style F fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    style G fill:#E0F2F1,stroke:#009688,stroke-width:2px
    style H fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
```

### Format Detection

```mermaid
graph TD
    A[ Data File] --> B[ Sample Lines]
    B --> C[ Try JSONL]
    B --> D[ Try Arrow]
    B --> E[ Try Parquet]

    C --> F[ Score Confidence]
    D --> F
    E --> F

    F --> G[ Select Best Format]
    G --> H[ Load with Format]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style C fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style D fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style E fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style G fill:#FFD54F,stroke:#F57C00,stroke-width:3px
    style H fill:#81C784,stroke:#388E3C,stroke-width:3px
```

---

## Adaptive Learning Rate 

### LR Adjustment Flow

```mermaid
graph TD
    A[ Training Step] --> B{⏱ Check Interval?}
    B -->|No| C[ Continue]
    B -->|Yes| D[ Compute Loss Window]

    D --> E{ Improvement?}
    E -->|Yes| F[ Update Best Loss]
    E -->|No| G[ Increment Plateau Counter]

    F --> H[ Reset Counter]
    H --> C

    G --> I{⏳ Counter > Patience?}
    I -->|No| C
    I -->|Yes| J{ Gradients Stable?}

    J -->|Yes| K[ Boost LR +15%]
    J -->|No| L[ Reduce LR -30%]

    K --> M[ Apply New LR]
    L --> M
    M --> N[ Reset Counter]
    N --> C

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style F fill:#81C784,stroke:#388E3C,stroke-width:2px
    style G fill:#FFCC80,stroke:#EF6C00,stroke-width:2px
    style K fill:#66BB6A,stroke:#2E7D32,stroke-width:3px
    style L fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style C fill:#80DEEA,stroke:#00838F,stroke-width:2px
```

### LR Schedule Visualization

```mermaid
graph LR
    A[ Step 0] --> B[ Warmup Phase]
    B --> C[ Peak LR]
    C --> D[ Cosine Decay]
    D --> E[ Restart 1]
    E --> F[ Peak LR]
    F --> G[ Cosine Decay]
    G --> H[ Final LR]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFAB91,stroke:#D84315,stroke-width:2px
    style C fill:#FFD54F,stroke:#F57C00,stroke-width:3px
    style D fill:#90CAF9,stroke:#1565C0,stroke-width:2px
    style E fill:#CE93D8,stroke:#8E24AA,stroke-width:2px
    style F fill:#FFD54F,stroke:#F57C00,stroke-width:3px
    style G fill:#90CAF9,stroke:#1565C0,stroke-width:2px
    style H fill:#81C784,stroke:#388E3C,stroke-width:3px
```

---

## RLHF Training 

### PPO Training Loop

```mermaid
graph TD
    A[ Start RLHF] --> B[ Load Policy Model]
    B --> C[ Load Reward Model]
    C --> D[ For Each Epoch]

    D --> E[ Sample Prompts]
    E --> F[ Generate Responses]
    F --> G[ Compute Rewards]
    G --> H[ Compute Advantages]

    H --> I[ PPO Update Loop]
    I --> J[ Compute Ratio π_new/π_old]
    J --> K[ Clip Ratio ε=0.2]
    K --> L[ Policy Loss]
    L --> M[ Value Loss]
    M --> N[ Total Loss]

    N --> O[ Backward]
    O --> P[ Optimizer Step]
    P --> Q{ KL < Threshold?}

    Q -->|No| R[ Adjust KL Penalty]
    Q -->|Yes| S{ More Updates?}
    R --> S

    S -->|Yes| I
    S -->|No| T{ More Epochs?}
    T -->|Yes| D
    T -->|No| U[ Save Final Model]

    style A fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style F fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style G fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style I fill:#2196F3,stroke:#1565C0,stroke-width:3px,color:#fff
    style K fill:#FFCC80,stroke:#EF6C00,stroke-width:2px
    style U fill:#81C784,stroke:#388E3C,stroke-width:3px
```

---

## Evaluation & Testing 

### Evaluation Flow

```mermaid
graph LR
    A[ Eval Trigger] --> B[ Set Eval Mode]
    B --> C[ Disable Dropout]
    C --> D[ For Each Val Batch]
    D --> E[ Forward Pass]
    E --> F[ Compute Loss]
    F --> G[ Accumulate Metrics]
    G --> H{ More Batches?}
    H -->|Yes| D
    H -->|No| I[ Compute Averages]
    I --> J[ Log Metrics]
    J --> K[ Set Train Mode]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style E fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style I fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style K fill:#81C784,stroke:#388E3C,stroke-width:3px
```

### Generation Quality Testing

```mermaid
graph TD
    A[ Test Prompts] --> B[ Generate Responses]
    B --> C[ Compute Perplexity]
    B --> D[ Compute Distinct-2]
    B --> E[ Compute Coherence]
    B --> F[ Compute Repetition Rate]

    C --> G[ Aggregate Metrics]
    D --> G
    E --> G
    F --> G

    G --> H{ Quality Good?}
    H -->|Yes| I[ Continue Training]
    H -->|No| J[ Adjust Hyperparams]
    J --> I

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:3px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style E fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    style F fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    style G fill:#E0F2F1,stroke:#009688,stroke-width:2px
    style H fill:#FFCC80,stroke:#EF6C00,stroke-width:3px
    style I fill:#81C784,stroke:#388E3C,stroke-width:3px
    style J fill:#FFAB91,stroke:#D84315,stroke-width:2px
```

---

## Multi-GPU Training 

### Distributed Training Flow

```mermaid
graph LR
    A[ Rank 0] --> E[ Broadcast Params]
    B[ Rank 1] --> E
    C[ Rank 2] --> E
    D[ Rank 3] --> E

    E --> F[ Forward Pass All]
    F --> G[ Backward Pass All]
    G --> H[ All-Reduce Gradients]
    H --> I[ Optimizer Step All]
    I --> J[ Synchronized Parameters]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    style D fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style E fill:#FFCC80,stroke:#EF6C00,stroke-width:3px
    style H fill:#90CAF9,stroke:#1565C0,stroke-width:3px
    style J fill:#81C784,stroke:#388E3C,stroke-width:3px
```

### DeepSpeed ZeRO Stages

```mermaid
graph TD
    A[ Standard DP] --> B[ All GPUs: Full Model Copy]

    C[ ZeRO-1] --> D[ Shard Optimizer States]

    E[ ZeRO-2] --> F[ Shard Optimizer + Gradients]

    G[ ZeRO-3] --> H[ Shard All States]

    B --> I[ High Memory Usage]
    D --> J[ Lower Memory]
    F --> K[ Even Lower Memory]
    H --> L[ Lowest Memory]

    style A fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    style C fill:#FFF9C4,stroke:#F57C00,stroke-width:2px
    style E fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    style G fill:#80DEEA,stroke:#00838F,stroke-width:2px
    style I fill:#EF5350,stroke:#B71C1C,stroke-width:2px,color:#fff
    style J fill:#FFD54F,stroke:#F57C00,stroke-width:2px
    style K fill:#81C784,stroke:#388E3C,stroke-width:2px
    style L fill:#4DD0E1,stroke:#00838F,stroke-width:3px
```

---

## Error Recovery 

### Error Handling Flow

```mermaid
graph TD
    A[ Error Detected] --> B{ Error Type?}

    B -->| NaN Loss| C[⏭ Skip Step & Log]
    B -->| Grad Explosion| D[ Reduce LR & Clip]
    B -->| OOM| E[ Clear Cache & Reduce Batch]
    B -->| Data Error| F[⏭ Skip File & Continue]
    B -->| Network Error| G[ Retry Connection]

    C --> H{ Frequency?}
    D --> H
    E --> H
    F --> H
    G --> H

    H -->| Rare| I[ Resume Training]
    H -->| Frequent| J[ Adjust Config]
    J --> I

    style A fill:#EF5350,stroke:#B71C1C,stroke-width:3px,color:#fff
    style C fill:#FFCC80,stroke:#EF6C00,stroke-width:2px
    style D fill:#FFAB91,stroke:#D84315,stroke-width:2px
    style E fill:#CE93D8,stroke:#8E24AA,stroke-width:2px
    style F fill:#90CAF9,stroke:#1565C0,stroke-width:2px
    style G fill:#80CBC4,stroke:#00695C,stroke-width:2px
    style I fill:#81C784,stroke:#388E3C,stroke-width:3px
```

---

## Legend 

### Color Coding System

```mermaid
graph LR
    A[ Input/Start] --> B[ Processing]
    B --> C[ Computation]
    C --> D[ Success/Output]

    E[ Warning] --> F[ Critical/Error]

    style A fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    style B fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    style C fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    style D fill:#C8E6C9,stroke:#388E3C,stroke-width:3px
    style E fill:#FFF9C4,stroke:#F57C00,stroke-width:2px
    style F fill:#FFCDD2,stroke:#C62828,stroke-width:2px
```

### Icon Legend

-  Start/Launch
-  Loop/Cycle
-  Success/Complete
-  Error/Failure
-  Warning
-  Metrics/Data
-  Model/Intelligence
-  Settings/Config
-  Storage/Memory
-  Check/Verify
-  Output/Result
-  Target/Goal
-  Increase/Up
-  Decrease/Down
-  Tool/Optimization
-  Fast/Expert
-  Combination
-  Explosion/Critical
-  Network/Distributed

---

## Summary

These beautiful, color-coded flowcharts provide clear visualizations of all major Ava training framework components with an intuitive color scheme:

- **Blue** : Input, initialization, and starting points
- **Orange** : Processing and transformation steps
- **Purple** : Computation and analysis
- **Green** : Success, completion, and outputs
- **Yellow** : Warnings and attention points
- **Red** : Errors and critical situations

Each flowchart uses emojis for quick visual recognition and makes complex training workflows easy to understand at a glance! 
