#!/usr/bin/env python3
"""
Check syntax of all new Python files without importing them.
"""

import ast
import sys
from pathlib import Path

def check_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "✅ Syntax OK"
    except SyntaxError as e:
        return False, f"❌ Syntax Error: {e}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def main():
    files_to_check = [
        "code/src/Ava/training/shardformer_integration.py",
        "code/src/Ava/training/distributed_optimizers.py",
        "code/src/Ava/training/colossalai_enhanced_features.py",
        "code/scripts/test_colossalai_enhanced.py",
        "code/scripts/validate_colossalai_imports.py",
    ]

    print("=" * 80)
    print("Checking Syntax of New Colossal-AI Files")
    print("=" * 80)

    all_ok = True
    for filepath in files_to_check:
        full_path = Path(filepath)
        if full_path.exists():
            ok, msg = check_syntax(full_path)
            print(f"{filepath:60s} {msg}")
            if not ok:
                all_ok = False
        else:
            print(f"{filepath:60s} ❌ File not found")
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("✅ All files have valid syntax!")
        return 0
    else:
        print("❌ Some files have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
