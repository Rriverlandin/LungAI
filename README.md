# LungAI

AI-assisted pneumonia classification from chest X-rays.

LungAI is a Building AI final project and working research prototype. It uses a
fine-tuned MobileNetV2 model to classify an X-ray as `NORMAL` or `PNEUMONIA`.
It is **not a medical device, diagnostic system, or substitute for a clinician**.

## 1. Your idea in a nutshell

Chest X-ray → preprocessing → MobileNetV2 → pneumonia score → class prediction.

The project explores whether transfer learning can support the initial review of
chest X-rays while keeping the model's limitations visible. The output is a
classification score for research and education, not a diagnosis.

## 2. Background

Reviewing medical images requires expertise, and mistakes have different costs.
In this task, a false negative means that a pneumonia image is classified as
normal; a false positive means that a normal image is flagged as pneumonia.
LungAI therefore reports per-class precision, recall, F1, and a confusion matrix
instead of presenting accuracy alone.

## 3. Data and AI techniques

The prototype uses the public **Chest X-Ray Images (Pneumonia)** dataset, arranged
as `NORMAL` and `PNEUMONIA` directories. Local data is excluded from Git to avoid
publishing a large dataset copy.

- Input: 224 × 224 grayscale X-rays
- Architecture: MobileNetV2 transfer learning
- Preprocessing: grayscale-to-RGB and MobileNetV2 rescaling inside the model
- Regularization: augmentation, dropout, early stopping, and class weights
- Fine-tuning: final 30 MobileNetV2 layers with a low learning rate
- Decision threshold: 0.60, selected on a held-out validation subset

| Split | Images |
|---|---:|
| Training | 4,695 |
| Validation | 521 |
| Test | 624 |
| Total | 5,840 |

## 4. How is it used?

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a single-image prediction from the repository root:

```bash
python -m src.predict path/to/xray.jpeg
```

Evaluate the saved model on `NORMAL/` and `PNEUMONIA/` folders:

```bash
python -m src.evaluate data/chest_xray/test
```

The complete experiment history is in
[`notebooks/01_first_test.ipynb`](notebooks/01_first_test.ipynb).

## 5. Results and challenges

Final V4 performance on the untouched 624-image test set:

| Metric | Result |
|---|---:|
| Accuracy | **87.18%** |
| Macro F1 | **85.32%** |
| NORMAL precision / recall / F1 | 95.83% / 68.80% / 80.10% |
| PNEUMONIA precision / recall / F1 | 83.99% / 98.21% / 90.54% |

![V4 confusion matrix](notebooks/models/V4_final_confusion_matrix.png)

![V4 classification metrics](notebooks/models/V4_final_metrics.png)

High pneumonia recall comes with lower normal recall: the model flags many
pneumonia cases, but also produces false positives. Other limitations include
dataset imbalance, possible dataset-specific artifacts, a single-source test set,
no external clinical validation, and no probability calibration. The score must
not be interpreted as a clinical probability.

## 6. What next?

- Validate on independent, multi-institutional data.
- Audit demographic and acquisition-device bias.
- Add calibration, confidence intervals, and patient-level split checks.
- Review errors with qualified radiology professionals.
- Add explainability only after validating that it is clinically useful.
- Package a privacy-aware demo with out-of-distribution rejection.

## 7. Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Architecture: MobileNetV2 through TensorFlow/Keras
- Course: [Building AI by Elements of AI](https://buildingai.elementsofai.com/)

## Repository structure

```text
LungAI/
├── notebooks/                 # Experiment notebook, plots, saved V4 model
├── src/                       # Inference and evaluation commands
├── tests/                     # Lightweight inference contract tests
├── models/                    # Base-weight notes; downloaded weights ignored
├── data/                      # Local dataset; ignored by Git
├── requirements.txt
└── README.md
```

## Reproducibility note

The saved model is provided for inference. The notebook records the exploratory
training process, but exact retraining can vary by hardware and implementation.
Reported final metrics use threshold `0.60` on the fixed test set.
