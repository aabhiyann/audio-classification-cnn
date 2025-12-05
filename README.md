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

<<<<<<< HEAD
## Project Structure
=======
## Project Structure

- `data/`
  - `dog/` – WAV files labeled as dog
  - `cat/` – WAV files labeled as cat
  - `bird/` – WAV files labeled as bird
- `notebooks/`
  - `01_explore_audio.ipynb` – EDA on waveforms and Mel-spectrograms; visual comparison of classes
  - `02_cnn_baseline.ipynb` – baseline CNN training + evaluation on Mel-spectrogram "images"
  - `03_cnn_improved.ipynb` – improved CNN experiments (capacity reduction, regularization) on small dataset
  - `04_cnn_full_data.ipynb` – final experiments on full dataset with comprehensive evaluation
  - `05_transfer_learning.ipynb` – transfer learning with YAMNet embeddings; comparison with CNN models
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

## Baseline Results (Small Dataset)

From early runs on a small subset of the data (using the explicit train/val/test split in `02_cnn_baseline.ipynb`):

- **Training accuracy** ≈ **0.89**
- **Validation accuracy** ≈ **0.60**
- **Test accuracy** ≈ **0.42** on a small held-out set.

These numbers indicate that:

- The CNN can fit the training data reasonably well.
- It performs clearly above random guessing (1/3) on validation/test.
- There is some overfitting, which is expected given the relatively small dataset and simple regularization.

---

## Final Model and Results

We trained several CNN architectures on Mel-spectrograms of animal sounds (dog / cat / bird). Using the full dataset (610 audio clips) and a stratified train/val/test split (440 / 78 / 92), our best model is:

- **CNN + Dropout(0.3)**  
  Conv(32) → MaxPool → Conv(64) → MaxPool → Flatten → Dense(64, ReLU) → Dropout(0.3) → Dense(3, Softmax)

**Test set performance (on 92 held-out clips):**

- Accuracy ≈ **88%**
- Macro F1 ≈ **0.88**
- Balanced performance across all three classes

Compared to the baseline CNN without Dropout, the regularized model achieves higher test accuracy and lower test loss, and reduces overfitting. Earlier experiments on a tiny 60-sample subset showed that strong Dropout (0.5) actually hurt performance, highlighting that regularization becomes effective only when enough training data is available.

See `04_cnn_full_data.ipynb` for complete results, confusion matrices, and detailed analysis.

### Transfer Learning Results (YAMNet)

We also experimented with transfer learning using pre-trained YAMNet embeddings (`05_transfer_learning.ipynb`):

- **YAMNet + Dense Head**: Test accuracy ≈ **62%**, Macro F1 ≈ **0.62**

**Comparison of all models:**

| Model                  | Test Accuracy | Test Loss |  Macro F1 | Notes                |
| ---------------------- | ------------: | --------: | --------: | -------------------- |
| Baseline CNN           |        83.70% |    0.6283 |     ~0.81 | Trained from scratch |
| **CNN + Dropout(0.3)** |    **88.04%** |    0.5503 | **~0.88** | **Best model**       |
| YAMNet + Dense Head    |        61.96% |    0.8990 |     ~0.62 | Transfer learning    |
   
**Key finding**: Training a CNN from scratch on Mel-spectrograms outperformed transfer learning with YAMNet for this specific task. This demonstrates that transfer learning is not always better—it depends on the task, dataset size, and domain alignment. See `05_transfer_learning.ipynb` for detailed analysis.

---

## Current Status (Deliverable II) & Next Steps

- **Status**

  - Data downloaded and organized into `dog/`, `cat/`, `bird/` folders under `data/`.
  - Mel-spectrogram pipeline implemented and validated in `01_explore_audio.ipynb`.
  - Baseline CNN implemented, trained, and evaluated with explicit train/val/test splits in `02_cnn_baseline.ipynb`.
  - Initial baseline on a **small 60-sample subset**: train accuracy ≈ 0.89, validation accuracy ≈ 0.60, test accuracy ≈ 0.42.
  - **Experiment 1** (capacity reduction) in `03_cnn_improved.ipynb`: reducing the Dense layer from 64 → 32 halved parameters but did **not** improve generalization on the tiny dataset.
  - **Experiment 2** (Dropout on small dataset) in `03_cnn_improved.ipynb`: strong Dropout (0.5) further hurt performance, showing that heavy regularization + very little data leads to underfitting.
  - **Full-data experiments completed** in `04_cnn_full_data.ipynb`: trained baseline CNN and CNN+Dropout(0.3) on all 610 clips (440 train / 78 val / 92 test). The Dropout model is our final chosen model with ≈88% test accuracy.
  - **Transfer learning experiments completed** in `05_transfer_learning.ipynb`: used pre-trained YAMNet to extract embeddings and trained a classifier head. YAMNet achieved ≈62% test accuracy, confirming that CNN from scratch is better for this task.

- **Key Learnings from Initial Experiments**

  - Starting with a small subset (60 samples) was useful for debugging and validating the pipeline.
  - Baseline CNN + Mel-spectrogram preprocessing is correct and learns meaningful features.
  - With only 9 validation samples (3 per class), validation metrics are extremely noisy.
  - Simply reducing model capacity (Dense 32) does not guarantee better generalization.
  - Strong Dropout with very small data can hurt performance more than it helps.
  - Scaling to the full dataset (610 clips) plus moderate Dropout (0.3) significantly improves test accuracy and gives stable metrics.

- **Immediate next steps**

  - Refactor common code (data loading, preprocessing, model definitions) from notebooks into `src/` modules.
  - Prepare final project presentation summarizing all experiments and findings.
  - Optionally explore simple data augmentation (time shift, additive noise) to test robustness.
  - Optionally save final model weights for reproducibility.

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
