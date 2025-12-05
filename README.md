## Animal Sound Classifier

This project is for **CSCI 6366: Neural Networks & Deep Learning** at The George Washington University.

We build deep learning models to classify animal sounds (**dog**, **cat**, **bird**) from short audio clips using:

- Mel-spectrograms and 2D Convolutional Neural Networks (CNNs)
- Hybrid **CRNN** architectures (CNN + GRU)
- Sequence models based on **Vision Transformers (ViT)** over spectrogram "images"
- Transfer learning with pre-trained audio models such as **YAMNet**

The goal is to build a clean, reproducible pipeline and compare a simple CNN baseline against more advanced architectures.

---

## Team Members

- Shambhavi Adhikari (G37903602) — GitHub: `@Shambhaviadhikari`
- Rakshitha Mamilla (G23922354) — GitHub: `@M-Rakshitha`
- Abhiyan Sainju (G22510509) — GitHub: `@aabhiyann`

---

## Dataset

We use the **Human Words Audio Classification** dataset (Kaggle):

https://www.kaggle.com/datasets/chiragchhaya/human-words-audio-classification

Each audio file is labeled as:

- `dog`
- `cat`
- `bird`

Properties:

- Mono `.wav` file
- Automatically resampled to **16 kHz**
- ~1 second duration
- Converted into **128×128 Mel-spectrograms**

---

## Project Structure