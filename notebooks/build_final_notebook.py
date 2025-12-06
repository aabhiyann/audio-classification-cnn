#!/usr/bin/env python3
"""
Script to build the FINAL submission notebook by combining sections from multiple notebooks.
"""

import json
from pathlib import Path

def load_notebook(path):
    """Load a Jupyter notebook."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    """Save a Jupyter notebook."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def create_title_cell():
    """Create the title cell for the FINAL notebook."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏆 FINAL PROJECT SUBMISSION: Animal Sound Classification\n",
            "\n",
            "**Course:** CSCI 6366 - Neural Networks & Deep Learning  \n",
            "**Team:**\n",
            "- Shambhavi Adhikari (G37903602)\n",
            "- Rakshitha Mamilla (G23922354)\n",
            "- Abhiyan Sainju (G22510509)\n",
            "\n",
            "---\n",
            "\n",
            "## Executive Summary\n",
            "\n",
            "This notebook presents our complete audio classification project, systematically exploring multiple approaches to classify animal sounds (dog, cat, bird) from short audio clips.\n",
            "\n",
            "**Best Result:** CNN + Dropout(0.3) achieving **88.04% test accuracy**\n",
            "\n",
            "**Key Finding:** Training CNNs from scratch on task-specific Mel-spectrogram features (88.04%) significantly outperforms transfer learning with YAMNet (66%) by 22 percentage points.\n",
            "\n",
            "### Project Structure:\n",
            "\n",
            "1. **Data Exploration** - Understanding the dataset and audio features\n",
            "2. **Baseline CNN** - Initial model development and validation\n",
            "3. **Full Dataset Training** - Scaling up and improving with regularization\n",
            "4. **Alternative Features** - MFCC-based 1D CNN approach\n",
            "5. **Transfer Learning** - YAMNet pre-trained model experiments\n",
            "6. **Final Analysis** - Complete comparison and key insights\n",
            "\n",
            "**Dataset:** 610 mono WAV files (~1 second, 16 kHz)\n",
            "- Dog: 210 samples\n",
            "- Cat: 207 samples\n",
            "- Bird: 193 samples\n",
            "\n",
            "---"
        ]
    }

def create_section_header(title, description=""):
    """Create a section header cell."""
    source = [f"# {title}\n"]
    if description:
        source.append(f"\n{description}\n")
    source.append("\n---\n")
    
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

# Paths
notebooks_dir = Path(".")
output_path = notebooks_dir / "FINAL_project_submission.ipynb"

# Load source notebooks
print("Loading source notebooks...")
nb_01 = load_notebook(notebooks_dir / "01_explore_audio.ipynb")
nb_02 = load_notebook(notebooks_dir / "02_cnn_baseline.ipynb")
nb_04 = load_notebook(notebooks_dir / "04_cnn_full_data.ipynb")
nb_04_comp = load_notebook(notebooks_dir / "04-comparingthemodels.ipynb")
nb_06 = load_notebook(notebooks_dir / "06_transfer_learning_yamnet_embeddings.ipynb")

print("Building FINAL notebook...")

# Start with empty notebook structure
final_nb = {
    "cells": [],
    "metadata": nb_04["metadata"],  # Use metadata from base notebook
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add title
final_nb["cells"].append(create_title_cell())

# PART 1: EDA from notebook 01
print("  Adding Part 1: Data Exploration...")
final_nb["cells"].append(create_section_header(
    "PART 1: Data Exploration & Understanding",
    "First, let's explore our dataset to understand the audio characteristics of each class."
))
# Add cells from notebook 01 (skip first title cell)
for cell in nb_01["cells"][1:]:
    final_nb["cells"].append(cell)

# PART 2: Baseline CNN from notebook 02
print("  Adding Part 2: Baseline CNN...")
final_nb["cells"].append(create_section_header(
    "PART 2: Baseline CNN Model",
    "Now we build our initial CNN model to establish a baseline performance."
))
# Add all cells from notebook 02 (skip first title cell)
for cell in nb_02["cells"][1:]:
    final_nb["cells"].append(cell)

# PART 3: Full Dataset from notebook 04
print("  Adding Part 3: Full Dataset Training...")
final_nb["cells"].append(create_section_header(
    "PART 3: Full Dataset Training & Regularization",
    "Scale up to the complete dataset (610 clips) and compare baseline vs regularized models."
))
# Add all cells from notebook 04 (skip first title cell)
for cell in nb_04["cells"][1:]:
    final_nb["cells"].append(cell)

# PART 4: MFCC 1D CNN from notebook 04-comparing
print("  Adding Part 4: MFCC Alternative...")
final_nb["cells"].append(create_section_header(
    "PART 4: Alternative Feature Representation - MFCC",
    "Explore MFCC features with 1D CNN as an alternative to Mel-spectrograms."
))

# Extract relevant cells from 04-comparing (look for MFCC and 1D CNN related cells)
# We'll take cells that are about MFCC extraction and 1D CNN training
mfcc_cells_added = False
for i, cell in enumerate(nb_04_comp["cells"]):
    if cell["cell_type"] == "markdown":
        source_text = "".join(cell["source"]).lower()
        # Look for MFCC-related sections
        if any(keyword in source_text for keyword in ["mfcc", "1d cnn", "mel-frequency"]):
            final_nb["cells"].append(cell)
            mfcc_cells_added = True
    elif cell["cell_type"] == "code" and mfcc_cells_added:
        source_code = "".join(cell["source"])
        # Add code cells related to MFCC or 1D CNN
        if any(keyword in source_code for keyword in ["mfcc", "MFCC", "Conv1D", "1D CNN"]):
            final_nb["cells"].append(cell)
            # Stop after we've added some MFCC-related code
            if len([c for c in final_nb["cells"][-10:] if c["cell_type"] == "code"]) >= 5:
                break

# PART 5: Transfer Learning from notebook 06
print("  Adding Part 5: Transfer Learning...")
final_nb["cells"].append(create_section_header(
    "PART 5: Transfer Learning with YAMNet",
    "Apply transfer learning using pre-trained YAMNet model and compare with from-scratch training."
))
# Add all cells from notebook 06 (skip first title cell)
for cell in nb_06["cells"][1:]:
    final_nb["cells"].append(cell)

# PART 6: Final Comparison (new synthesis)
print("  Adding Part 6: Final Comparison...")
final_nb["cells"].append(create_section_header(
    "PART 6: Complete Analysis & Key Findings",
    "Synthesize all results and identify key insights from our experiments."
))

# Add comparison cells
final_nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "from matplotlib.patches import Patch\n",
        "\n",
        "# Complete results from all experiments\n",
        "results_data = {\n",
        "    'Model': [\n",
        "        'CNN + Dropout(0.3)',\n",
        "        '1D CNN (MFCC)',\n",
        "        'Baseline CNN',\n",
        "        'YAMNet (Full Sequence)',\n",
        "        'YAMNet (Averaged)'\n",
        "    ],\n",
        "    'Accuracy (%)': [88.04, 84.43, 83.70, 66.0, 61.96],\n",
        "    'Input Type': [\n",
        "        'Mel-spectrogram (2D)',\n",
        "        'MFCC (1D)',\n",
        "        'Mel-spectrogram (2D)',\n",
        "        'Raw waveform',\n",
        "        'Raw waveform'\n",
        "    ],\n",
        "    'Approach': [\n",
        "        'From scratch',\n",
        "        'From scratch',\n",
        "        'From scratch',\n",
        "        'Transfer learning',\n",
        "        'Transfer learning'\n",
        "    ],\n",
        "    'Notebook Section': [\n",
        "        'Part 3',\n",
        "        'Part 4',\n",
        "        'Part 3',\n",
        "        'Part 5',\n",
        "        'Part 5'\n",
        "    ]\n",
        "}\n",
        "\n",
        "results_df = pd.DataFrame(results_data)\n",
        "results_df = results_df.sort_values('Accuracy (%)', ascending=False).reset_index(drop=True)\n",
        "results_df.insert(0, 'Rank', range(1, len(results_df) + 1))\n",
        "\n",
        "print(\"\\n\" + \"=\"*90)\n",
        "print(\"🏆 COMPLETE MODEL COMPARISON\")\n",
        "print(\"=\"*90)\n",
        "print(results_df.to_string(index=False))\n",
        "print(\"=\"*90)\n",
        "print(f\"\\nBest Model: {results_df.iloc[0]['Model']} - {results_df.iloc[0]['Accuracy (%)']}% accuracy\")\n",
        "print(f\"Performance range: {results_df['Accuracy (%)'].max() - results_df['Accuracy (%)'].min():.2f} percentage points\")"
    ]
})

final_nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visualization: Model comparison\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
        "\n",
        "# Left: All models\n",
        "colors = ['#2ecc71' if 'scratch' in app else '#e74c3c' for app in results_df['Approach']]\n",
        "colors[0] = '#f39c12'  # Highlight best\n",
        "\n",
        "bars = ax1.barh(results_df['Model'], results_df['Accuracy (%)'], color=colors, alpha=0.8)\n",
        "bars[0].set_edgecolor('black')\n",
        "bars[0].set_linewidth(2)\n",
        "\n",
        "for i, acc in enumerate(results_df['Accuracy (%)']):\n",
        "    ax1.text(acc + 1, i, f'{acc:.2f}%', va='center', fontweight='bold')\n",
        "\n",
        "ax1.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')\n",
        "ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')\n",
        "ax1.set_xlim(0, 100)\n",
        "ax1.grid(axis='x', alpha=0.3)\n",
        "\n",
        "legend = [\n",
        "    Patch(facecolor='#f39c12', edgecolor='black', label='🏆 Best Model'),\n",
        "    Patch(facecolor='#2ecc71', alpha=0.8, label='Train from Scratch'),\n",
        "    Patch(facecolor='#e74c3c', alpha=0.8, label='Transfer Learning')\n",
        "]\n",
        "ax1.legend(handles=legend, loc='lower right')\n",
        "\n",
        "# Right: Transfer learning vs from scratch\n",
        "approaches = ['Transfer Learning\\n(Best)', 'From Scratch\\n(Best)']\n",
        "accs = [66.0, 88.04]\n",
        "bars2 = ax2.bar(approaches, accs, color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)\n",
        "\n",
        "for bar, acc in zip(bars2, accs):\n",
        "    height = bar.get_height()\n",
        "    ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{acc:.2f}%',\n",
        "            ha='center', va='bottom', fontsize=14, fontweight='bold')\n",
        "\n",
        "ax2.annotate('', xy=(1, 88.04), xytext=(0, 66.0),\n",
        "            arrowprops=dict(arrowstyle='<->', color='black', lw=2))\n",
        "ax2.text(0.5, 77, '+22%', ha='center', fontsize=12, fontweight='bold',\n",
        "        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))\n",
        "\n",
        "ax2.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')\n",
        "ax2.set_title('Transfer Learning vs Training from Scratch', fontsize=14, fontweight='bold')\n",
        "ax2.set_ylim(0, 100)\n",
        "ax2.grid(axis='y', alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n🎯 Key Insight: Training from scratch on task-specific features\")\n",
        "print(\"   significantly outperforms transfer learning for this specialized task!\")"
    ]
})

final_nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Key Findings Summary\n",
        "\n",
        "### 1. Best Model Performance\n",
        "- **CNN + Dropout(0.3)**: 88.04% test accuracy\n",
        "- Macro F1: 0.88\n",
        "- Balanced performance across all three classes\n",
        "\n",
        "### 2. Training from Scratch > Transfer Learning\n",
        "- From scratch (88.04%) beats transfer learning (66%) by **22 percentage points**\n",
        "- Why? **Domain mismatch**: YAMNet trained on generic AudioSet vs our specific animal vocalizations\n",
        "- Lesson: Pre-trained models need domain alignment, not just capacity\n",
        "\n",
        "### 3. Feature Representation Matters\n",
        "- Mel-spectrograms (2D): 88%\n",
        "- MFCC (1D): 84%\n",
        "- YAMNet embeddings: 66%\n",
        "- Insight: 2D representations provide richer information for CNNs\n",
        "\n",
        "### 4. Regularization Impact\n",
        "- Baseline CNN: 83.7%\n",
        "- + Dropout(0.3): 88.04%\n",
        "- Improvement: +4.3 percentage points\n",
        "- Critical for preventing overfitting with moderate datasets\n",
        "\n",
        "### 5. Dataset Size Sweet Spot\n",
        "- 610 clips sufficient for training from scratch\n",
        "- Transfer learning typically helps more with <100 samples\n",
        "- Our dataset in the \"sweet spot\" for from-scratch training\n",
        "\n",
        "---\n",
        "\n",
        "## Conclusions\n",
        "\n",
        "This project demonstrates that:\n",
        "\n",
        "1. **Task-specific training can outperform transfer learning** when:\n",
        "   - Dataset size is moderate (500-1000 samples)\n",
        "   - Domain is specialized (not covered by pre-trained model)\n",
        "   - Good features can be engineered (Mel-spectrograms)\n",
        "\n",
        "2. **Systematic experimentation reveals insights:**\n",
        "   - Started with baseline (83.7%)\n",
        "   - Improved with regularization (88.04%)\n",
        "   - Tested alternatives (MFCC: 84%)\n",
        "   - Compared with SOTA (YAMNet: 66%)\n",
        "   - Understood WHY each approach performed as it did\n",
        "\n",
        "3. **Simple architectures with proper regularization work best**\n",
        "   - 2-layer CNN + Dropout beats complex models\n",
        "   - Match model complexity to dataset size\n",
        "\n",
        "### Final Result\n",
        "**🏆 88.04% Test Accuracy with CNN + Dropout(0.3)**\n",
        "\n",
        "### Team\n",
        "- Shambhavi Adhikari (G37903602)\n",
        "- Rakshitha Mamilla (G23922354)\n",
        "- Abhiyan Sainju (G22510509)\n",
        "\n",
        "**Course:** CSCI 6366 - Neural Networks & Deep Learning  \n",
        "**The George Washington University**"
    ]
})

# Save the final notebook
print(f"\\nSaving FINAL notebook to: {output_path}")
save_notebook(final_nb, output_path)

print(f"✅ Done! Created notebook with {len(final_nb['cells'])} cells")
print("\\nNotebook structure:")
print("  - Part 1: Data Exploration (from notebook 01)")
print("  - Part 2: Baseline CNN (from notebook 02)")
print("  - Part 3: Full Dataset Training (from notebook 04)")
print("  - Part 4: MFCC Alternative (from notebook 04-comparing)")
print("  - Part 5: Transfer Learning (from notebook 06)")
print("  - Part 6: Final Comparison & Conclusions (new synthesis)")

