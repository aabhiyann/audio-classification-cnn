# Presentation Outline: Animal Sound Classification with CNNs

**Course:** CSCI 6366 (Neural Networks and Deep Learning)  
**Project:** Audio Classification using CNN  
**Team:** Shambhavi Adhikari, Rakshitha Mamilla, Abhiyan Sainju

---

## Slide 1: Title Slide
- **Title**: Animal Sound Classification using Convolutional Neural Networks
- **Subtitle**: Comparing CNN Training from Scratch vs Transfer Learning
- **Team Members**: Shambhavi Adhikari, Rakshitha Mamilla, Abhiyan Sainju
- **Course**: CSCI 6366 - Neural Networks and Deep Learning
- **Date**: [Presentation Date]

---

## Slide 2: Problem & Dataset
- **Problem**: Classify animal sounds into three categories: **dog**, **cat**, **bird**
- **Dataset**:
  - Source: Human Words Audio Classification (Kaggle)
  - **610 audio clips** total (210 dog, 207 cat, 193 bird)
  - Mono `.wav` files, ~1 second duration
  - Resampled to 16 kHz for processing
- **Challenge**: Learn discriminative features from audio waveforms

---

## Slide 3: Preprocessing Pipeline
- **Step 1**: Load audio at 16 kHz (mono)
- **Step 2**: Compute **Mel-spectrogram**
  - Parameters: `n_fft=1024`, `hop_length=512`, `n_mels=128`
  - Convert to dB scale
- **Step 3**: Fix input size
  - Crop or pad to **128×128** (mel bands × time frames)
- **Step 4**: Normalize to [0, 1] and add channel dimension
- **Result**: Each audio clip → `(128, 128, 1)` "image" for CNN input

---

## Slide 4: Baseline CNN Architecture
- **Architecture**:
  - Input: `(128, 128, 1)` Mel-spectrogram
  - Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
  - Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
  - Flatten → Dense(64, ReLU) → Dense(3, Softmax)
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical crossentropy
  - Batch size: 8-16, Epochs: 10-20
- **Initial Results** (small 60-sample subset):
  - Train acc: ~89%, Val acc: ~60%, Test acc: ~42%

---

## Slide 5: Small Dataset Experiments
- **Experiment 1: Capacity Reduction**
  - Reduced Dense layer: 64 → 32 units (~50% fewer parameters)
  - **Result**: Did not improve; validation accuracy decreased (60% → 40%)
  - **Lesson**: Simply reducing capacity doesn't guarantee better generalization

- **Experiment 2: Dropout Regularization**
  - Added Dropout(0.5) to baseline
  - **Result**: Performance worsened (test acc: 42% → 33%)
  - **Lesson**: Strong regularization + very little data → underfitting

- **Key Insight**: With only 9 validation samples, metrics are extremely noisy

---

## Slide 6: Full Dataset Experiments - Setup
- **Scale up**: Use all **610 audio clips**
- **Stratified train/val/test split**:
  - Train: 440 samples (151 dog, 150 cat, 139 bird)
  - Validation: 78 samples (27 dog, 26 cat, 25 bird)
  - Test: 92 samples (32 dog, 31 cat, 29 bird)
- **Two models compared**:
  1. Baseline CNN (Dense 64, no regularization)
  2. CNN + Dropout(0.3) (regularized)

---

## Slide 7: Full Dataset Results - Training Curves
- **Baseline CNN**:
  - Train acc: 100%, Val acc: ~95%
  - Shows overfitting (perfect training, lower validation)

- **CNN + Dropout(0.3)**:
  - Train acc: ~99%, Val acc: ~92%
  - Better generalization (smaller train-val gap)

- **Visualization**: Training curves showing loss and accuracy over epochs

---

## Slide 8: Full Dataset Results - Test Performance
- **Baseline CNN**:
  - Test accuracy: **83.70%**
  - Test loss: 0.6283
  - Macro F1: ~0.81

- **CNN + Dropout(0.3)** (Best Model):
  - Test accuracy: **88.04%** ✅
  - Test loss: 0.5503
  - Macro F1: ~0.88
  - Balanced performance across all classes (F1 > 0.88 for each)

- **Confusion Matrix**: Show actual confusion matrix visualization

---

## Slide 9: Transfer Learning with YAMNet
- **Approach**:
  1. Use pre-trained **YAMNet** (trained on AudioSet)
  2. Extract **1024-D embeddings** from each audio waveform
  3. Train small classifier: Dense(128, ReLU) + Dropout(0.3) → Dense(3, Softmax)

- **Advantages**:
  - Faster training (only train classifier head)
  - Leverages knowledge from large-scale AudioSet dataset

- **Results**:
  - Test accuracy: **61.96%**
  - Test loss: 0.8990
  - Macro F1: ~0.62

---

## Slide 10: Model Comparison
| Model | Test Accuracy | Test Loss | Macro F1 | Training Approach |
|-------|--------------:|----------:|---------:|-------------------|
| Baseline CNN | 83.70% | 0.6283 | ~0.81 | Train from scratch |
| **CNN + Dropout(0.3)** | **88.04%** | 0.5503 | **~0.88** | **Train from scratch** ✅ |
| YAMNet + Head | 61.96% | 0.8990 | ~0.62 | Transfer learning |

**Key Finding**: CNN from scratch outperforms transfer learning for this task!

---

## Slide 11: Why CNN Outperformed YAMNet
1. **Domain Mismatch**:
   - YAMNet trained on generic AudioSet events
   - Not specialized for dog/cat/bird sounds
   - CNN learns task-specific features directly

2. **Dataset Size**:
   - 610 clips sufficient for training from scratch
   - Transfer learning benefits more with very small datasets

3. **Task Alignment**:
   - Mel-spectrograms capture discriminative features for animal sounds
   - YAMNet embeddings may miss these specific patterns

---

## Slide 12: Key Takeaways
1. **Regularization matters**: Dropout(0.3) improved test accuracy from 84% → 88%
2. **Dataset size matters**: Small dataset (60 samples) → noisy metrics; full dataset (610) → reliable results
3. **Transfer learning isn't always better**: Depends on task, dataset size, and domain alignment
4. **Empirical evaluation is crucial**: Test multiple approaches, don't assume transfer learning will win

---

## Slide 13: Project Structure & Reproducibility
- **Notebooks**:
  - `01_explore_audio.ipynb` - EDA and preprocessing validation
  - `02_cnn_baseline.ipynb` - Initial CNN baseline
  - `03_cnn_improved.ipynb` - Architecture experiments
  - `04_cnn_full_data.ipynb` - Final CNN results
  - `05_transfer_learning.ipynb` - YAMNet comparison

- **Reproducibility**:
  - Fixed random seeds (42)
  - Same train/val/test splits across notebooks
  - Local YAMNet model (no internet required)
  - Clear documentation in README

---

## Slide 14: Future Work & Conclusions
- **Future Work**:
  - Code refactoring into `src/` modules
  - Data augmentation (time shift, noise)
  - Explore other transfer learning models (VGGish)
  - Larger dataset experiments

- **Conclusions**:
  - Successfully built CNN classifier achieving **88% test accuracy**
  - Demonstrated importance of regularization and dataset size
  - Showed that training from scratch can outperform transfer learning for domain-specific tasks
  - Clean, reproducible pipeline ready for further experiments

---

## Slide 15: Questions & Thank You
- **Questions?**
- **Thank you!**

---

## Notes for Presenters:
- **Slide 7**: Show actual training curve plots from notebook 04
- **Slide 8**: Include confusion matrix visualization
- **Slide 9**: Show YAMNet architecture diagram if helpful
- **Slide 10**: Emphasize the comparison table - this is a key finding
- **Slide 11**: This is where you demonstrate critical thinking about results
- **Keep it concise**: Aim for ~15 minutes presentation, ~5 minutes Q&A

