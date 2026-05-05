---
title: SVM Explorer
emoji: 🧠
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

<div align="center">

# 🧠 ML Explorer

**An interactive tool for learning, visualizing, and experimenting with Support Vector Machines**

[![Python](https://img.shields.io/badge/Python-3.9%2B-4fc3f7?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00e5ff?style=for-the-badge)](LICENSE)

[🚀 Live Demo on Hugging Face](https://huggingface.co/spaces/YOUR_USERNAME/svm-explorer) &nbsp;|&nbsp;
[📦 GitHub Repo](https://github.com/YOUR_USERNAME/svm-explorer)

</div>

---

## ✨ Features

| Tab | What you get |
|-----|-------------|
| 📖 **What is SVM?** | Visual explanation of hyperplanes, margins, support vectors & the kernel trick |
| 🎯 **Decision Boundary** | Live plot that updates as you tune `C`, `kernel`, and `gamma` |
| 🔬 **Kernel Comparison** | Side-by-side view of all 4 kernels on the same dataset |
| 📊 **Model Evaluation** | Confusion matrix, classification report, and C vs accuracy curves |

### 📂 Upload Your Own Data
Drop any **CSV file** and pick which columns to use as features and label — the app handles the rest.

> **Requirements:** at least 2 numeric columns + 1 binary label column (values `0` and `1`)

---

## 🖥️ Preview

```
┌─────────────────────────────────────────────────────┐
│  🧠 SVM Explorer                                     │
│  Support Vector Machine — Interactive Learning Tool  │
├──────────────┬──────────────────────────────────────┤
│  ⚙️ Settings │  📖 What is SVM?  🎯 Boundary  ...   │
│              │                                       │
│  Kernel RBF  │   [live decision boundary plot]       │
│  C = 1.00    │                                       │
│  γ = scale   │   Accuracy  Kernel  C   Sup.Vecs      │
│              │   97.5%     RBF    1.0    24          │
│  Dataset:    │                                       │
│  ● Built-in  │                                       │
│  ○ Upload CSV│                                       │
└──────────────┴──────────────────────────────────────┘
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/svm-explorer.git
cd svm-explorer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501` 🎉

---

## ☁️ Deploy to Hugging Face Spaces

**Option A — Import from GitHub (recommended)**

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Click **"Import from GitHub"** and paste your repo URL
4. Hit **Create Space** — done ✅

**Option B — Push manually**

```bash
pip install huggingface_hub
git clone https://huggingface.co/spaces/YOUR_USERNAME/svm-explorer hf-space
cd hf-space
cp /path/to/{app.py,requirements.txt,README.md} .
git add . && git commit -m "🧠 SVM Explorer" && git push
```

---

## 🎛️ Tunable Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Kernel** | `linear` / `rbf` / `poly` / `sigmoid` | The kernel function |
| **C** | 0.01 → 20 | Regularization strength |
| **Gamma** | `scale` / `auto` / custom | RBF & Poly kernel width |
| **Degree** | 2 → 6 | Polynomial kernel degree |
| **Test Size** | 10% → 50% | Train/test split ratio |
| **Random State** | 0 → 100 | Reproducibility seed |

---

## 📁 Project Structure

```
svm-explorer/
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## 🔬 How SVM Works (TL;DR)

```
Data points → Kernel maps to high-D space → SVM finds max-margin hyperplane
                                                        ↓
                                           Only support vectors matter
                                                        ↓
                                           Predict new points by side of boundary
```

---

## 📦 Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
seaborn>=0.12.0
```

---

<div align="center">

Made with ❤️ · [Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org)

</div>
