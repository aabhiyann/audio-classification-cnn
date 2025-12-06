#!/usr/bin/env python3
"""
Fix imports properly in the FINAL notebook.
"""

import json
from pathlib import Path

# Load notebook
nb_path = Path('FINAL_project_submission.ipynb')
nb = json.load(open(nb_path))

print("Fixing redundant imports...")

# Cell 3 - remove DATA_DIR, keep the rest
cell3_source = ''.join(nb['cells'][3]['source'])
if 'DATA_DIR = Path' in cell3_source:
    # Remove the DATA_DIR line
    lines = cell3_source.split('\n')
    new_lines = [l for l in lines if 'DATA_DIR = Path' not in l]
    nb['cells'][3]['source'] = ['\n'.join(new_lines)]
    print("  Cell 3: Removed DATA_DIR redefinition")

# Cells to completely replace with comment
cells_to_replace = [23, 25, 41, 44, 58, 80]
for i in cells_to_replace:
    if i < len(nb['cells']):
        source = ''.join(nb['cells'][i]['source']).strip()
        # Only replace if it's ONLY imports/config
        lines = [l.strip() for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]
        is_only_imports = all(
            any(kw in l for kw in ['import ', 'from ', 'DATA_DIR', 'CLASS_NAMES', 'label_to_index', 'SAMPLE_RATE'])
            for l in lines if l
        )
        if is_only_imports or not lines:
            nb['cells'][i]['source'] = ['# ✅ All imports and configuration defined in PART 1\n']
            print(f"  Cell {i}: Replaced with comment")

# Cell 81 - MFCC configuration that overrides main config
cell81_source = ''.join(nb['cells'][81]['source'])
if 'SAMPLE_RATE = 22050' in cell81_source:
    # This is the MFCC section with different parameters
    # Replace with a note about using different params for MFCC
    nb['cells'][81]['source'] = [
        '# Note: MFCC section uses different parameters optimized for MFCC extraction\n',
        '# These override the global settings for this section only\n',
        'SAMPLE_RATE_MFCC = 22050  # Different from main (16000)\n',
        'DURATION_MFCC = 3  # seconds\n',
        'N_MFCC_FEATURES = 40\n'
    ]
    print(f"  Cell 81: Updated MFCC-specific configuration")

# Save
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n✅ Done! All imports centralized in PART 1")

