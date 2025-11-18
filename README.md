# Animal Sound Classifier

This project is for **CSCI 4/6366: Intro to Deep Learning** at The George Washington University.  
We are building a deep learning model to classify animal sounds (**dog**, **cat**, **bird**) from audio clips using:

- Mel-spectrograms and 2D Convolutional Neural Networks (CNNs)
- Transfer learning with pre-trained audio models such as **YAMNet** or **VGGish**

---

## Team Members

- Shambhavi Adhikari (G37903602)— GitHub: @Shambhaviadhikari
- Rakshitha Mamilla (G23922354)— GitHub: @M-Rakshitha
- Abhiyan Sainju (G22510509) — GitHub: @aabhiyann

---

## Dataset

We use the following dataset(s):

- **Human Words Audio Classification (Kaggle)**  
  Link: https://www.kaggle.com/datasets/chiragchhaya/human-words-audio-classification

Each audio file is a WAV file labeled as one of three classes:
- `dog`
- `cat`
- `bird`

---

## Project Structure (initial)

Planned structure:

- `notebooks/` – Jupyter notebooks for EDA, preprocessing, and model experiments.
- `src/` – Python scripts for data loading, preprocessing, and model training.
- `data/` – (Git-ignored for large raw data; instructions will be provided to download locally.)

At this stage (Deliverable II), the repo contains:
- `README.md` – project summary, team info, dataset links
- `notebooks/initial_exploration.ipynb` **or** `src/initial_pipeline.py` – initial project code

---

## Goals

1. Train a baseline CNN model on Mel-spectrograms of the audio clips.
2. Improve performance using transfer learning with YAMNet/VGGish embeddings.
3. Evaluate accuracy, confusion matrices, and analyze where the model struggles.

## Initial Work

- Dataset downloaded and examined.
- Mel-spectrogram generation tested.
- Initial notebook/Python file created to begin preprocessing and baseline CNN setup.

## Future Work

- CNN-based audio classification pipeline built on Mel-spectrograms.
- Include a minimal baseline model and data/EDA checklist.
- Add an optional Phase-2 using pretrained audio embeddings (YAMNet/VGGish).
- Report results using Accuracy, Macro-F1, and a confusion matrix.

