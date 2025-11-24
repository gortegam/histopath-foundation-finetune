# 🧬 Histopath Foundation Fine‑Tune  
### Deep Learning for Tumor vs Normal Classification in H&E‑Stained Colorectal Tissue

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-red)
![Vision](https://img.shields.io/badge/Model-ResNet18-green)
![ML](https://img.shields.io/badge/Task-Histopathology-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

# 📌 Executive Summary

This project fine‑tunes a deep learning model (ResNet18) on **H&E‑stained colorectal histopathology patches** to classify **tumor** vs **normal** tissue.  
It demonstrates an end‑to‑end medical imaging ML workflow:

- Dataset preprocessing  
- Transfer learning with PyTorch  
- Custom training loops  
- Model evaluation (ROC AUC 0.9956)  
- Grad‑CAM interpretability  
- Streamlit inference app  

This is the type of workflow used in real-world digital pathology AI pipelines (e.g., for frozen sections, tumor detection, second‑reader systems).

---

# 🧠 TL;DR (For Recruiters)

> **A full medical imaging ML pipeline:** Fine‑tuned a CNN on colorectal H&E patches, achieved ROC AUC **0.9956**, built Grad‑CAM interpretability, and deployed an interactive Streamlit inference app.

This project shows:

- Applied ML on medical images  
- Deep learning proficiency (PyTorch)  
- Ability to build clinician‑interpretable tools  
- Domain knowledge as a histotechnician  

---

# 🧩 Skills Demonstrated

### **Deep Learning & Vision**
- Transfer learning (ResNet18)
- Custom PyTorch training loops  
- Dataloaders, augmentations, and batching  
- ROC AUC, confusion matrix, probability calibration  

### **Model Interpretability**
- Grad‑CAM heatmaps  
- Attention visualization  
- Tumor vs normal morphological reasoning  

### **Deployment & Tooling**
- Streamlit app for real‑time inference  
- Automated Grad‑CAM overlay during prediction  
- Clean project structure + reproducibility  

### **Healthcare/Pathology Domain**
- Understanding of H&E morphology  
- Binary tumor vs normal detection  
- Digital pathology workflow alignment  

---

# 🎯 Project Structure

```
histopath-foundation-finetune/
│
├── src/
│   ├── train.py                 # training loop + scheduler + optimizer
│   ├── evaluate.py              # confusion matrix, ROC AUC
│   ├── gradcam.py               # Grad‑CAM implementation
│   ├── dataset.py               # PyTorch dataset for CRC patches
│   └── utils.py                 # helpers for logging + preprocessing
│
├── notebooks/
│   └── histopath_workflow.ipynb # EDA + model tests (optional)
│
├── runs/
│   ├── confusion_matrix.png
│   └── gradcam_example.png
│
├── images/
│   ├── streamlit_demo.png
│   └── gradcam_overlay.png
│
├── app.py                       # Streamlit inference UI
├── requirements.txt
└── README.md
```

---

# 🧪 Dataset

| Property    | Value |
|------------|-------|
| Source     | CRC‑VAL‑HE‑7K (Zenodo: 1214456) |
| Images     | H&E‑stained colorectal patches |
| Task       | TUM (Tumor) vs NORM (Normal) |
| Resolution | 224×224 |
| Train/Test | ~400/50/50 per class |

To download the dataset:

```bash
wget https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip
unzip CRC-VAL-HE-7K.zip
```

---

# 🚀 Training the Model

### 1. Activate environment
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python3 -m src.train     --epochs 20     --batch-size 32     --learning-rate 1e-4     --train-dir data/train     --val-dir data/val     --save-dir runs/
```

This fine‑tunes a ResNet18 initialized with ImageNet weights.

---

# 📈 Results

| Metric   | Value |
|---------|--------|
| **ROC AUC** | **0.9956** |
| Accuracy | ~96–99% (from confusion matrix) |

Example confusion matrix and ROC curves are saved in `/runs/`.

---

# 🔍 Grad‑CAM Interpretability

Grad‑CAM visualizations highlight the pixels most responsible for classifying “tumor” vs “normal.”

Example output:

- **Tumor patch:** Strong activation around crowded nuclei, hyperchromasia, loss of polarity  
- **Normal patch:** Minimal activation except at glandular boundaries  

Images live in `/images/gradcam_overlay.png` and in the README preview section.

---

# 🖥️ Streamlit Inference App

Run:

```bash
streamlit run app.py
```

Features:

- Upload an H&E patch  
- Model outputs **class + probability**  
- Grad‑CAM overlay auto‑generated  
- Clinician‑friendly UI  

Screenshot stored under `images/streamlit_demo.png`.

---

# 🧬 Why This Project Matters for Employers

This project mirrors *exactly* what ML engineers or data scientists do in healthcare AI teams:

- Fine‑tuning neural networks on imaging data  
- Applying domain‑specific augmentations  
- Using Grad‑CAM for clinical interpretability  
- Deploying lightweight inference tools for clinicians  

This project proves you can contribute to:

- Computational pathology  
- Radiology AI  
- Frozen section decision support  
- Tumor detection / screening tools  

---

# 🧪 How to Reproduce

1. Download dataset  
2. Install dependencies  
3. Run `src/train.py`  
4. Evaluate with `src/evaluate.py`  
5. Launch Streamlit app  

All code is deterministic (`torch.manual_seed(42)`).

---

# 🗣️ How I'd Explain This in an Interview

> “I fine‑tuned a ResNet18 on colorectal H&E patches to classify tumor vs normal tissue.  
> The model achieved ROC AUC 0.9956.  
> I implemented Grad‑CAM so pathologists can see which regions influenced the prediction, which is essential for trust.  
> Finally, I wrapped it in a Streamlit app to create a real‑world inference tool.  
> This pipeline is very similar to what computational pathology and radiology AI teams build in production.”

---

# 🔮 Future Work

- SHAP for image explainability  
- Add ViT (Vision Transformer) model variant  
- Multi‑class extension (CRC subtypes)  
- Frozen‑section real‑time workflow  
- MLOps: ONNX export + FastAPI inference  

---

# 📬 Contact

**Giancarlo Ortega**  
📍 Cedar Rapids, Iowa  
GitGitHub: https://github.com/gortegam  
LinkedIn: *your link here*  
Email: *your email here*
