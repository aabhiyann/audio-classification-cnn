#!/usr/bin/env python3
"""
Clean up redundant imports and configuration from the FINAL notebook.
"""

import json
from pathlib import Path

def is_import_or_config_line(line):
    """Check if a line is an import or config line"""
    l = line.strip()
    if not l or l.startswith('#'):
        return True
    keywords = [
        'import ', 'from ', 'DATA_DIR =', 'CLASS_NAMES =', 'label_to_index =', 
        'random.seed', 'tf.random.set_seed', 'warnings.', 'SAMPLE_RATE =', 
        'N_FFT =', 'HOP_LENGTH =', 'N_MELS =', 'TEST_SIZE =', 'VAL_SIZE =', 
        'MAX_FRAMES =', 'N_MFCC =', 'DURATION =', 'MAX_FILES_PER_CLASS =',
        'print(f"Data directory:', 'print(f"Sample rate:', 'print(f"Max embedding'
    ]
    return any(kw in l for kw in keywords)

# Load notebook
nb_path = Path('FINAL_project_submission.ipynb')
nb = json.load(open(nb_path))

print("Cleaning up redundant imports and configuration...")

cells_modified = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and i > 5:  # Skip the main import cell
        source = ''.join(cell['source'])
        lines = source.split('\n')
        
        # Check if it has imports/config
        if any('import' in l or 'DATA_DIR =' in l or 'CLASS_NAMES =' in l 
               or 'random.seed' in l or 'tf.random.set_seed' in l for l in lines):
            
            # Filter out import/config lines
            cleaned_lines = [l for l in lines if not is_import_or_config_line(l)]
            remaining = '\n'.join(cleaned_lines).strip()
            
            # If nothing meaningful remains, replace with comment
            if not remaining or remaining == '':
                cell['source'] = ['# ✅ All imports and configuration defined in PART 1\n']
                cells_modified += 1
                print(f"  Cell {i}: Replaced with comment")
            elif remaining != source.strip():
                # Keep the non-import code
                cell['source'] = [remaining + '\n']
                cells_modified += 1
                print(f"  Cell {i}: Removed imports, kept other code")

# Save cleaned notebook
print(f"\nSaving cleaned notebook...")
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Done! Modified {cells_modified} cells")
print(f"   All imports and configuration now centralized in PART 1")

