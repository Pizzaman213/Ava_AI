# Claude Documentation - Ava LLM Training Framework

This directory contains all Claude-generated documentation for the Ava project.

## 📁 Documentation Structure

### Training & Configuration
- **[TRAINING_FIXES.md](TRAINING_FIXES.md)** - Critical training fixes (learning rate optimization)
- **[CONFIG_CHANGES_SUMMARY.md](CONFIG_CHANGES_SUMMARY.md)** - Configuration change log
- **[QUICK_FIX.txt](QUICK_FIX.txt)** - Quick reference for common fixes

### Optimization Guides
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Comprehensive optimization guide
- **[TRAINING_OPTIMIZATIONS_APPLIED.md](TRAINING_OPTIMIZATIONS_APPLIED.md)** - Applied optimizations summary
- **[OPTIMIZATIONS.md](OPTIMIZATIONS.md)** - Optimization overview

### Codebase Management
- **[CODEBASE_CLEANUP_SUMMARY.md](CODEBASE_CLEANUP_SUMMARY.md)** - Codebase cleanup summary
- **[ENHANCED_DATA_PREP_SUMMARY.md](ENHANCED_DATA_PREP_SUMMARY.md)** - Data preparation enhancements

## 🔗 Quick Links

### Main Project Documentation
- [Main README](../README.md) - Project overview
- [Claude.md](../code/Claude.md) - AI change log
- [Dev Log](../code/dev_log.md) - Development log

### Configuration Files
- [GPU Configs](../code/configs/gpu/) - GPU training configurations
- [CPU Configs](../code/configs/cpu/) - CPU training configurations

### Training Scripts
- [Train Script](../code/scripts/training/train.py) - Main training script
- [Diagnostic Tools](../code/scripts/diagnose_training.py) - Training diagnostics

## 📝 Documentation Guidelines for Claude

**IMPORTANT**: All new Claude-generated documentation MUST be placed in `/project/claude_docs/`

When creating new documentation:
1. **Location**: Always place in `/project/claude_docs/` directory
2. **Naming**: Use descriptive filenames (UPPER_CASE_WITH_UNDERSCORES.md)
3. **Index**: Add entry to this README
4. **Change Log**: Update `/project/code/Claude.md` with the change
5. **Header**: Include date and purpose in document header
6. **Cross-reference**: Link to related docs where applicable

## 🔄 Recent Updates

**2025-10-06**
- Added memory silent mode configuration
- Organized all docs into dedicated folder
- Created this README

---

**Maintained by**: Ava Development Team
**Last Updated**: 2025-10-06

### Memory Management
- **[MEMORY_SILENT_MODE.md](MEMORY_SILENT_MODE.md)** - Suppress memory warnings (silent_mode flag)

