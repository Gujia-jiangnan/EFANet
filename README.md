# **Evidence-Focused Attention Network for Interpretable Fine-Grained Food Analysis**
This repository provides the official implementation of **EFANet**, an evidence‑focused attention network designed for **interpretable fine‑grained food image recognition**.  

EFANet explicitly models **class-discriminative visual evidence** and produces **faithful, localized explanations** while maintaining competitive recognition performance.

![intro](C:\Users\Yi\repo\可解释\latex\intro.jpg)

---

## 🔍 Overview

Fine-grained food recognition often suffers from:
- reliance on contextual cues (e.g., side dishes),
- incomplete or over-smoothed visual explanations,
- limited faithfulness between explanations and predictions.

EFANet addresses these issues by:
- introducing an **Active Feature Retrieval Mechanism**,
- achieving **representation decoupling**,
- producing **faithful, class-specific explanations**.

The method is evaluated on **Food‑101** and **FoodX‑251**, achieving strong performance in both **classification accuracy** and **explanation faithfulness** (Insertion / Deletion metrics).

---

## 📊 Experimental Results

### Classification Performance
EFANet achieves competitive Top‑1 and Top‑5 accuracy on Food‑101 and FoodX‑251 while preserving interpretability.

### Explanation Faithfulness
Compared with Grad‑CAM, Attention Rollout, Prompt‑CAM, and LRP, EFANet produces:
- lower deletion scores,
- higher insertion scores,
- better localization of fine‑grained food attributes.

---

## 🧩 Visualization Examples

EFANet highlights **class-defining attributes** while suppressing irrelevant regions, demonstrating superior interpretability in multi-instance and fine‑grained scenarios.

---

## 🚀 Installation

### Requirements
```bash
Python >= 3.11
PyTorch == 2.2.1
timm
numpy
opencv-python
matplotlib
tqdm
scikit-learn
pandas
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

### Supported Datasets
- **Food‑101**
- **FoodX‑251**

Expected directory structure:
```text
dataset/
├── food101/
│   ├── train/
│   └── val/
└── foodx251/
    ├── train/
    └── val/
```

---

## 🔥 Training

Example training command:
```bash
./distributed_train.sh 1 --data-dir ../dataset/food101/ --model efanet_small_patch16_224 --pretrained -b 128 --epochs 10 --opt adamw --lr 3e-5 --sched cosine --warmup-epochs 2 --warmup-lr 1e-5 --min-lr 1e-6 --weight-decay 1e-2 -j 8 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --num-classes 101

./distributed_train.sh 1 --data-dir ../dataset/food101/ --model efanet_base_patch16_224 --pretrained -b 128 --epochs 10 --opt adamw --lr 3e-5 --sched cosine --warmup-epochs 2 --warmup-lr 1e-5 --min-lr 1e-6 --weight-decay 1e-2 -j 8 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --num-classes 101

./distributed_train.sh 1 --data-dir ../dataset/food101/ --model efanet_large_patch16_224 --pretrained -b 64 --epochs 10 --opt adamw --lr 3e-5 --sched cosine --warmup-epochs 2 --warmup-lr 1e-5 --min-lr 1e-6 --weight-decay 1e-2 -j 8 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --drop-path 0.2 --num-classes 101
```

Outputs will be saved to:

```text
./outputs
```

---

## 🔎 Evaluation

```bash
python eval.py \
--data-dir ../dataset/food101/ \
--model efanet_base_patch16_224 \
--checkpoint efa_base.tar \
-b 128 \
--num-classes 101
```

---

## 🧠 Evidence Visualization

Generate explanation maps:

See visual.ipynb and tools.py

---

## 📏 Faithfulness Metrics

EFANet supports:
- **Insertion**
- **Deletion**
- **Draw AUC Curves** (See draw.ipynb)

```bash
python evaluate_faithfulness.py
```
