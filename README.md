## Animal Sound Classifier

This project is for **CSCI 6366: Neural Networks & Deep Learning** at The George Washington University.

We are building a deep learning model to classify animal sounds (**dog**, **cat**, **bird**) from short audio clips using:

- Mel-spectrograms and 2D Convolutional Neural Networks (CNNs)
- Transfer learning with pre-trained audio models such as **YAMNet** or **VGGish**

The end goal is to build a clean, reproducible pipeline and compare a simple CNN baseline against transfer-learning based approaches.

---

## Team Members

- Shambhavi Adhikari (G37903602) — GitHub: `@Shambhaviadhikari`
- Rakshitha Mamilla (G23922354) — GitHub: `@M-Rakshitha`
- Abhiyan Sainju (G22510509) — GitHub: `@aabhiyann`

---

## Dataset

We use the following dataset:

- **Human Words Audio Classification (Kaggle)**  
  Link: https://www.kaggle.com/datasets/chiragchhaya/human-words-audio-classification

After filtering, each audio file is treated as belonging to one of three classes:

- `dog`
- `cat`
- `bird`

Each file is:

- A mono `.wav` clip
- Resampled to \(16{,}000\) Hz during preprocessing
- Roughly 1 second in duration (so they can be converted into fixed-size spectrograms)

---

## Project Structure

- `data/`
  - `dog/` – WAV files labeled as dog
  - `cat/` – WAV files labeled as cat
  - `bird/` – WAV files labeled as bird
- `notebooks/`
  - `01_explore_audio.ipynb` – EDA on waveforms and Mel-spectrograms; visual comparison of classes
  - `02_cnn_baseline.ipynb` – baseline CNN training + evaluation on Mel-spectrogram “images”
- `src/`
  - (planned) Python modules for reusable data loading, preprocessing, and model code
- `README.md` – this file

As the project matures, most of the notebook logic will be moved into `src/` for cleaner experiments and easier reproduction.

---

## Environment & Setup

- **Python**: 3.10+ recommended
- **Key packages**:
  - `tensorflow>=2.16,<3`
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

You can install dependencies (example with `pip`):

```bash
pip install "tensorflow>=2.16,<3" librosa numpy matplotlib scikit-learn
```

Place the dataset under `data/` with subfolders `dog/`, `cat/`, and `bird/` so that paths look like:

```text
audio-classification-cnn/
  data/
    dog/*.wav
    cat/*.wav
    bird/*.wav
```

Then you can open the notebooks in Jupyter or VS Code and run them end-to-end.

---

## Modeling Pipeline (Current Baseline)

At a high level, the current baseline pipeline does:

1. **Load audio**
   - Use `librosa.load(path, sr=16000)` to get a mono waveform at 16 kHz.
2. **Compute Mel-spectrogram**
   - Parameters: `n_fft=1024`, `hop_length=512`, `n_mels=128`, `power=2.0`.
   - Convert the power spectrogram to dB with `librosa.power_to_db`.
3. **Fix the input size**
   - Start with shape `(128, T)` (mel bands × time frames).
   - Center-crop or right-pad along the time axis to get `(128, 128)`.
4. **Normalize & reshape**
   - Per-example min–max normalization to \([0, 1]\).
   - Add a channel dimension → `(128, 128, 1)` (a grayscale “image”).
5. **Label encoding**
   - Map classes `["dog", "cat", "bird"]` to indices `[0, 1, 2]`.
   - Use one-hot vectors of length 3 for training.
6. **Train / val / test splits**
   - Stratified splits with `train_test_split` to create:
     - A **held-out test set** never seen during training.
     - Separate **train** and **validation** sets.
7. **CNN model (Keras)**
   - Input: `(128, 128, 1)`
   - Conv2D(32, 3×3, relu, same) → MaxPooling2D(2×2)
   - Conv2D(64, 3×3, relu, same) → MaxPooling2D(2×2)
   - Flatten → Dense(64, relu) → Dense(3, softmax)
   - Compile with `optimizer="adam"`, `loss="categorical_crossentropy"`, `metrics=["accuracy"]`.

All of this is currently implemented and demonstrated in `02_cnn_baseline.ipynb`.

---

## Current Baseline Results (Summary)

From early runs on a small subset of the data:

- **Training accuracy** climbs to roughly **0.85–0.90**.
- **Validation accuracy** stabilizes around **0.50–0.60**.
- **Test accuracy** on a held-out set is roughly **0.40–0.60** depending on the exact split and number of epochs.

These numbers indicate that:

- The CNN can fit the training data reasonably well.
- It performs clearly above random guessing (1/3) on validation/test.
- There is some overfitting, which is expected given the relatively small dataset and simple regularization.

As we move forward, we will add more systematic experiments, confusion matrices, and macro-F1 scores to better understand strengths and failure modes.

---

## Goals

1. Train a clear **baseline CNN** model on Mel-spectrograms of the audio clips.
2. Improve performance using transfer learning with **YAMNet/VGGish** embeddings.
3. Evaluate using **accuracy**, **macro-F1**, and **confusion matrices**, and analyze where the model struggles.

---

## Future Work & Roadmap

- **Code organization**
  - Move core logic (preprocessing, model definitions, training loops) from notebooks into `src/`.
  - Add simple configuration options for dataset paths, splits, and hyperparameters.
- **Transfer learning phase**
  - Extract embeddings from pre-trained audio models (YAMNet, VGGish).
  - Train shallow classifiers (e.g., MLP or small CNN heads) on top of those embeddings.
  - Compare against the Mel-spectrogram CNN baseline.
- **Evaluation & reporting**
  - Add confusion matrices and macro-F1.
  - Perform ablations (e.g., different input sizes, number of filters, data augmentation).
  - Summarize findings in a short written report / slides for the course deliverable.

This README will be updated as new experiments and results are added.
