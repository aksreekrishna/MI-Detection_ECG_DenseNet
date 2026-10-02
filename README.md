# BioPredictX: Residual-Guided Knowledge Distillation for Lightweight Myocardial Infarction Detection

BioPredictX is a deep learning framework engineered for accurate, real-time detection of Myocardial Infarction (MI) from Electrocardiogram (ECG) signals[cite: 1]. High-performance deep neural networks (like deep 1D CNNs, ResNet, or DenseNet variants) achieve excellent diagnostic accuracy but are computationally heavy, making them difficult to deploy on wearable devices or embedded clinical monitors. 

BioPredictX resolves this trade-off using **Residual-Guided Knowledge Distillation (KD)**[cite: 1]. By transferring rich representations and residual error corrections from a high-capacity teacher network into a streamlined student network, BioPredictX delivers high diagnostic precision with low latency and minimal memory footprint.

---

## 📌 Problem Statement & Key Idea

* **The Problem:** Early detection of Myocardial Infarction is critical to preventing irreversible cardiac damage and saving lives. Standard continuous ECG monitoring generates vast streams of time-series data, but accurate deep learning models often require significant memory and computing power, exceeding the limits of edge devices.
* **The Idea:** Train a complex, highly accurate **Teacher Model** on multi-lead ECG signals, then distill its knowledge into a lightweight **Student Model**. By incorporating **Residual-Guided Loss**, the student model learns not only the soft target outputs of the teacher, but also explicitly learns to compensate for residual feature discrepancies between the two networks.

---

## 🔄 Methodology & Workflow

```text
       ┌────────────────────────┐
       │   Raw ECG Datasets     │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Preprocessing Pipeline │  (Filtering, Segmentation, Normalization)
       └───────────┬────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌─────────────┐   ┌─────────────┐
   │ Train/Val   │   │ Test Split  │
   │    Split    │   └─────────────┘
   └──────┬──────┘
          │
          ├───► 1. Train Teacher Model (High Capacity)
          │
          └───► 2. Residual-Guided Knowledge Distillation
                      ├── Teacher Logits & Features
                      ├── Student Distillation Loss
                      └── Residual Correction Loss
                                │
                                ▼
                  3. Deploy Student Model on Edge Hardware
