# Quick Start: Data Processing

## 🎉 Auto-Organize is Now Default!

Just run the script - it automatically separates Q&A from web text:

```bash
python scripts/data_prep/prepare_data_rapids.py
```

**Result:**
- Q&A datasets → `/data/fine-tuning/` (for fine-tuning)
- Web text → `/data/processed/` (for pre-training)

---

## Common Usage

### Process All Datasets (Auto-Organized):
```bash
python scripts/data_prep/prepare_data_rapids.py
```

### Process Specific Datasets:
```bash
python scripts/data_prep/prepare_data_rapids.py \
  --datasets Open-Orca_OpenOrca meta-math_MetaMathQA
```

### Disable Auto-Organize (Old Behavior):
```bash
python scripts/data_prep/prepare_data_rapids.py --no-auto-organize
```

---

## What Gets Organized Where?

### → `/fine-tuning/` (Q&A Data):
- Open-Orca_OpenOrca
- OpenAssistant_oasst1 & oasst2
- meta-math_MetaMathQA
- CodeAlpaca-20k
- Python code instructions
- Medical flashcards
- And more Q&A datasets...

### → `/processed/` (Web Text):
- HuggingFaceFW_fineweb
- allenai_c4
- cc_news
- TinyStories
- And more general text...

---

## Need Help?

See full documentation:
- [DEMO_AUTO_ORGANIZE.md](DEMO_AUTO_ORGANIZE.md) - Complete guide
- [data/HOW_QA_DETECTION_WORKS.md](data/HOW_QA_DETECTION_WORKS.md) - How detection works
- [data/AUTO_ORGANIZE_FEATURE.txt](data/AUTO_ORGANIZE_FEATURE.txt) - Quick reference

