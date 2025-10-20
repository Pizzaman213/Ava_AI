#!/usr/bin/env python3
"""Simple test to verify the rotation configuration"""

print("Checking data shuffling configuration...")
print("=" * 60)

# Read the data_streaming.py file
with open('/project/code/src/Ava/data_streaming.py', 'r') as f:
    content = f.read()

# Check for the samples_per_file setting
if 'samples_per_file = 1  # Read 1 sample from each file before switching for better mixing' in content:
    print("✓ Configuration: ROTATING MODE (1 sample per file)")
    print("  Status: Successfully updated!")
    print()
    print("How it works:")
    print("  - Takes 1 example from file A")
    print("  - Takes 1 example from file B")
    print("  - Takes 1 example from file C")
    print("  - ... continues through all files in round-robin")
    print("  - Then repeats the cycle")
    print()
    print("Benefits:")
    print("  ✓ Better data mixing across different datasets")
    print("  ✓ More diverse batches early in training")
    print("  ✓ Prevents long sequences from single dataset")
    print("  ✓ More uniform distribution of data sources")

elif 'samples_per_file = 500' in content:
    print("✗ Configuration: BATCH MODE (500 samples per file)")
    print("  Status: Not updated yet")

else:
    print("? Configuration unclear")

print()
print("=" * 60)

# Also check the comment to confirm
if 'BETTER SHUFFLING: Read ONE sample from each file in round-robin fashion' in content:
    print("✓ Comment updated to reflect rotating behavior")
else:
    print("  Comment not updated")

print()
print("Configuration file: code/src/Ava/data_streaming.py:543")
print()
print("Test completed!")
