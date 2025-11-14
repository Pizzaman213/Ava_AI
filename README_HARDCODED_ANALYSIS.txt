================================================================================
HARDCODED VALUES ANALYSIS - COMPLETE REPORT SET
================================================================================

OVERVIEW
--------
This directory contains a comprehensive analysis of hardcoded values in the
Ava MoE training pipeline that should be configurable.

Total Issues Found: 102 hardcoded values
Critical Issues: 5 (must fix)
High/Medium Issues: 79 (should fix)
Low Issues: 18 (nice to have)

REPORT FILES
============

1. HARDCODED_VALUES_SUMMARY.txt (7.6 KB)
   - Executive summary of all findings
   - High-level overview by severity
   - Impact analysis (performance, portability, accuracy)
   - Recommended action phases with time estimates
   
   READ THIS FIRST if you want a quick overview.

2. HARDCODED_VALUES_QUICK_REFERENCE.txt (5.2 KB)
   - Quick lookup guide organized by severity
   - File locations and line numbers
   - Current values and what they control
   - Checklist for fixing issues
   - Configuration template for YAML
   
   READ THIS to find specific hardcoded values or when coding fixes.

3. HARDCODED_VALUES_ANALYSIS.md (27 KB)
   - Comprehensive detailed analysis
   - Complete tables with every hardcoded value
   - Explanations for why each should be configurable
   - Code snippets and exact locations
   - Recommendations with code examples
   - Testing strategy
   - Full impact analysis
   
   READ THIS for complete technical details and rationale.

CRITICAL ISSUES - WHAT TO FIX FIRST
====================================

1. EOS Token ID (Line 80 in moe_model.py)
   - Currently: eos_token_id: int = 3
   - Problem: Only works with specific tokenizers
   - Impact: Cannot use different tokenizers
   
2. Train/Val Split (Lines 735-740 in dataloader.py)
   - Currently: if file_hash < 85:  # 85% train
   - Problem: Cannot customize split ratios
   - Impact: Cannot balance datasets properly

3. Default Config Path (Lines 202, 615 in train.py)
   - Currently: "/project/code/configs/gpu/small.yaml"
   - Problem: Hardcoded full path, not portable
   - Impact: Only works on this specific system

4. Cache Directory (Line 496 in training_config.py)
   - Currently: "/tmp/difficulty_cache"
   - Problem: Uses /tmp (data lost on reboot)
   - Impact: Not portable, loses progress on restart

5. num_workers Override (Lines 1066-1072 in train.py)
   - Currently: Forced to 0 regardless of config
   - Problem: Ignores user settings for safety
   - Impact: 10-30% slower data loading

QUICK STATISTICS
================

By Severity:
  Critical: 5 issues (4-6 hours to fix)
  Medium:   79 issues (18-24 hours to fix)
  Low:      18 issues (2-4 hours to fix)

By File:
  dataloader.py ............ 32 issues (most affected)
  constants.py ............ 25 issues
  trainer.py ............. 16 issues
  training_config.py ...... 7 issues
  moe_model.py ........... 8 issues
  train.py ............... 6 issues
  Others ................. 8 issues

By Category:
  Magic Numbers ........... 45 issues (numeric literals)
  File Paths ............. 12 issues (hardcoded /tmp, /project)
  Device Specs ........... 8 issues (GPU/CPU selection)
  Logging Frequencies .... 8 issues (hardcoded intervals)
  Defaults ............... 10 issues (fallback values)
  Thresholds ............ 19 issues (memory, loss, performance)

GETTING STARTED
===============

Step 1: Choose Your Scope
  - Critical only? (4-6 hours) → Read QUICK_REFERENCE
  - Critical + High Priority? (12-18 hours) → Read SUMMARY
  - Everything? (20-30 hours) → Read ANALYSIS.md

Step 2: Understand the Issues
  - Open HARDCODED_VALUES_QUICK_REFERENCE.txt
  - Find the section for your target area
  - Note the file locations and line numbers

Step 3: Make Changes
  - Use constants.py as your example (it's well-structured)
  - Add YAML parameters for new configurable values
  - Update functions to accept config parameters
  - Test with different configuration values

Step 4: Document Your Changes
  - Update relevant config files with new options
  - Document in training_config.py dataclasses
  - Add to YAML template examples

CONFIGURATION APPROACH
======================

The codebase already has infrastructure for configuration:

1. constants.py - Centralized constants dataclasses
   - DataPipelineConstants (data loading)
   - TrainerConstants (training)
   - MoEConstants (expert routing)

2. training_config.py - Configuration management
   - TrainingConfigManager class
   - load_yaml_config() for YAML support
   - DynamicConfig for flexible config objects

3. YAML config files in /project/code/configs/
   - Define parameters by use case
   - Can override constants via config.constants section

RECOMMENDED APPROACH TO FIXING:

Phase 1 - Critical (4-6 hours):
  [] Fix EOS token ID configurability
  [] Fix train/val split configurability
  [] Fix default config path
  [] Fix /tmp directory references
  [] Add --force-workers flag

Phase 2 - High Priority (8-12 hours):
  [] Audit and document data loading constants
  [] Create YAML template for data loading
  [] Add constants validation function
  [] Update documentation

Phase 3 - Medium Priority (6-8 hours):
  [] Consolidate logging frequencies
  [] Make generation parameters configurable
  [] Improve device selection
  [] Document which constants are essential

Total Effort: ~20-30 hours for complete resolution

KEY INSIGHTS
============

GOOD NEWS:
- Infrastructure for configuration already exists
- Many parameters are already configurable
- Central constants.py is well-designed
- Most critical issues are quick fixes (1-5 lines each)

CHALLENGES:
- num_workers forced to 0 for safety (performance trade-off)
- Some defaults scattered across multiple files
- Not all code paths respect configuration overrides
- Some hardcoded values are "intentional" (like /tmp) but should be optional

IMPACT:
- Performance: 10-30% variation possible with different configurations
- Portability: Many hardcoded paths prevent running on different systems
- Flexibility: Cannot easily adapt to different datasets/models/GPUs
- Maintainability: Easier updates if all magic numbers are centralized

NEXT STEPS
==========

1. Prioritize by severity:
   - Must-fix critical issues first
   - Then high-priority for performance
   - Then low-priority for polish

2. Use existing infrastructure:
   - Follow patterns in constants.py
   - Use DataConfig/TrainingConfig classes
   - Add to YAML where appropriate

3. Test thoroughly:
   - Unit tests for config loading
   - Integration tests with different values
   - Performance tests to verify no regression
   - Portability tests across systems

4. Document well:
   - Add comments explaining configurability
   - Update YAML templates with new options
   - Create migration guide for users

For detailed guidance, see the appropriate report file above.

================================================================================
Report Generated: 2025-11-13
Analysis Includes: Data Loading, Training, Models, Logging, Device Config
Coverage: /project/code/src/Ava/ (all subdirectories) + /project/code/scripts/
================================================================================
