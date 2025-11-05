# Histopath Foundation Fine-Tune

Fine-tuning pretrained vision models for colorectal histopathology with reproducible training, evaluation, and interactive slide-level inference.

---

## 🔬 Overview

This project fine-tunes vision models (ResNet18 and optionally ViT) to classify colorectal histopathology patches as **Tumor (TUM)** or **Normal (NORM)**.  
It is designed to mirror real-world digital pathology workflows: dataset organization → model training → evaluation → visual interactive inference.

This repo demonstrates:

- End-to-end deep learning training pipeline using PyTorch
- Evaluation with ROC AUC and confusion matrix outputs
- Streamlit-based inference UI for clinicians and researchers
- Modular code structure suitable for extension or deployment

---

## 🧫 Dataset

| Property | Value |
|---------|-------|
| Source | **CRC-VAL-HE-7K (Zenodo: 1214456)** |
| Task | Binary classification (TUM vs NORM) |
| Image Type | H&E-stained tissue patches |
| Size | 224×224 per patch |
| Train/Val/Test | ~400 / 50 / 50 per class |

Dataset is **not included** in this repository.  
Download from: https://zenodo.org/record/1214456 and place into:

```
data/train/<class>/
data/val/<class>/
data/test/<class>/
```

---

## 🧠 Model & Training

| Component | Details |
|---------|---------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Loss | BCEWithLogitsLoss (binary) |
| Optimizer | AdamW |
| Device | CPU-compatible |
| Epochs | 1+ (demo), adjustable |

### Train
```bash
python -m src.train --data_dir data --epochs 5 --batch_size 8 --arch resnet18
```

### Evaluate
```bash
python -m src.evaluate --data_dir data --ckpt checkpoints/best.pt
```

Outputs:
- `runs/metrics.json`
- `runs/confusion_matrix.png`

---

## 📈 Results (Example Run)

| Metric | Value (demo) |
|--------|--------------|
| ROC AUC | ~0.99 |
| Accuracy | ~95–98% |
| Precision | >0.90 |
| Recall | >0.90 |

Confusion Matrix example:  
*(included in repo as `runs/confusion_matrix.png`)*

---

## 🖥️ Streamlit Interactive Inference App

### Launch
```bash
export PYTHONPATH=$(pwd)
streamlit run app/app.py -- --ckpt checkpoints/best.pt --labels_file runs/labels.json
```

### Example UI Output

![Streamlit Demo](images/streamlit_demo.png)

> Upload an H&E patch → model predicts probability of tumor vs normal → image + confidence shown live.

---

## 🗂️ Project Structure

```
histopath-foundation-finetune/
├── app/                    # Streamlit inference UI
├── src/                    # Training, evaluation, models, data loaders
├── runs/                   # metrics.json + confusion_matrix.png
├── checkpoints/            # model weights (ignored by git)
├── images/                 # screenshots for README
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚫 Files Not Included (By Design)

| Folder/File | Reason |
|-------------|--------|
| `/data` | Dataset is large; fetched externally |
| `/checkpoints` | Model weights > 40MB; regenerate locally |
| `.venv/` | Environment recreated from `requirements.txt` |

---

## 📄 License
MIT — Free to modify and use with attribution.

---

## 👤 Author
**Giancarlo Ortega**  
Histotechnician → Machine Learning Engineer (Biomedical AI)  
GitHub: https://github.com/gortegam  
LinkedIn: https://www.linkedin.com/in/giancarlo-ortega-8b051a2a6

---

> This project supports PathAI’s mission to improve diagnostic accuracy and impact patient outcomes through machine learning in clinical pathology.
