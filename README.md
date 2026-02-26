# 🧬 MolGuard — Akıllı İlaç Güvenlik Analiz Sistemi | Intelligent Drug Safety Analysis System

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"/></a>
  <a href="#"><img src="https://img.shields.io/badge/PyG-2.4+-green.svg" alt="PyG"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Colab-Pro-orange.svg" alt="Colab"/></a>
</p>

<p align="center">
  <b>Çok Görevli Grafik Sinir Ağları ile İlaç Güvenliği Tahmin Sistemi</b><br/>
  <b>Multi-Task Graph Neural Network for Drug Safety Prediction</b>
</p>

---

## 📖 İçindekiler | Table of Contents

- [Proje Hakkında | About](#-proje-hakkında--about)
- [Özellikler | Features](#-özellikler--features)
- [Mimari | Architecture](#-mimari--architecture)
- [Kurulum | Installation](#-kurulum--installation)
- [Veri Setleri | Datasets](#-veri-setleri--datasets)
- [Sonuçlar | Results](#-sonuçlar--results)
- [Web Uygulaması | Web Application](#-web-uygulaması--web-application)
- [Proje Yapısı | Project Structure](#-proje-yapısı--project-structure)

---

## 🎯 Proje Hakkında | About

**MolGuard**, ilaçların moleküler yapılarını Grafik Sinir Ağları (GNN) kullanarak analiz eden ve güvenlik tahminleri yapan kapsamlı bir derin öğrenme sistemidir.

**MolGuard** is a comprehensive deep learning system that analyzes drug molecular structures using Graph Neural Networks (GNN) to predict drug safety profiles.

### Problemin Tanımı | Problem Statement

İlaç güvenliği, sağlık sektörünün en kritik konularından biridir. Her yıl binlerce hasta, ilaçların yan etkilerinden ve ilaç-ilaç etkileşimlerinden zarar görmektedir. MolGuard, bu soruna yapay zeka tabanlı bir çözüm sunmaktadır.

Drug safety is one of the most critical issues in healthcare. Every year, thousands of patients suffer from adverse drug reactions and drug-drug interactions. MolGuard provides an AI-powered solution to this problem.

---

## ✨ Özellikler | Features

| Özellik / Feature | Açıklama / Description |
|---|---|
| 🔴 **Toksisite Tahmini / Toxicity Prediction** | İlacın toksik olup olmadığını tahmin eder / Predicts whether a drug is toxic |
| 🟡 **Yan Etki Tahmini / Side Effect Prediction** | Olası yan etkileri belirler / Identifies potential side effects |
| 🟢 **Etkinlik Tahmini / Efficacy Prediction** | İlacın etkinliğini değerlendirir / Evaluates drug efficacy |
| ⚠️ **İlaç Etkileşimi / Drug Interaction** | İki ilacın birlikte kullanım riskini tahmin eder / Predicts risk of combining two drugs |
| 💊 **Güvenli Alternatif / Safe Alternative** | Daha güvenli ilaç alternatifleri önerir / Suggests safer drug alternatives |
| 🧠 **Açıklanabilirlik / Explainability** | GNNExplainer ile tahmin nedenlerini gösterir / Shows prediction reasoning with GNNExplainer |
| 👤 **Hasta Profili / Patient Profile** | Hasta özelliklerine göre risk ayarlar / Adjusts risk based on patient characteristics |

---

## 🏗️ Mimari | Architecture

```
                         ┌──────────────────────┐
                         │   Hasta Profili       │
                         │   Patient Profile     │
                         └─────────┬─────────────┘
                                   │
┌─────────────────┐     ┌─────────▼──────────────┐     ┌──────────────────────┐
│ Moleküler Graf  │     │                        │     │    Çıktılar/Outputs: │
│ Molecular Graph │────▶│  Çok Görevli GNN       │────▶│  🔴 Toksisite        │
│ (atomlar/atoms  │     │  Multi-Task GNN        │     │  🟡 Yan Etkiler      │
│  + bağlar/bonds)│     │                        │     │  🟢 Etkinlik         │
└─────────────────┘     │  + Bilgi Grafiği       │     │  ⚠️ Etkileşimler     │
                        │    Knowledge Graph     │     │  💊 Alternatifler    │
┌─────────────────┐     │                        │     └──────────────────────┘
│ İlaç Etkileşim  │     │  + Hasta Koşullandırma │
│ DDI Graph       │────▶│    Patient Conditioning│
└─────────────────┘     └────────────┬───────────┘
                                     │
                        ┌────────────▼───────────┐
                        │    GNNExplainer        │
                        │    + Attention Viz      │
                        └────────────────────────┘
```

### Model Karşılaştırması | Model Comparison

| Model | Açıklama / Description |
|---|---|
| **GCN** | Graph Convolutional Network — Temel mimari / Baseline |
| **GAT** | Graph Attention Network — Dikkat mekanizmalı / With attention |
| **GIN** | Graph Isomorphism Network — En güçlü ayırt edici / Most powerful |

---

## ⚙️ Kurulum | Installation

### Google Colab (Önerilen / Recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/hasanoso151/MolGuard.git
%cd MolGuard
!pip install -r requirements.txt
```

---

## 📊 Veri Setleri | Datasets

| Veri Seti / Dataset | Açıklama / Description | Boyut / Size |
|---|---|---|
| **Tox21** | Toksisite / Toxicity | 7,831 bileşik |
| **SIDER** | Yan etkiler / Side effects | 1,427 ilaç |
| **BBBP** | Kan-beyin bariyeri / BBB permeability | 2,039 bileşik |
| **ClinTox** | Klinik toksisite / Clinical toxicity | 1,478 bileşik |
| **OGB-DDI** | İlaç etkileşimleri / Drug interactions | 1.3M kenar |
| **DRKG** | Bilgi grafiği / Knowledge graph | 5.8M ilişki |

---

## 📁 Proje Yapısı | Project Structure

```
MolGuard/
├── notebooks/                    # Colab Notebook'ları
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_explainability.ipynb
│   └── 06_drug_interaction.ipynb
├── src/
│   ├── data/                     # Veri işleme / Data processing
│   ├── models/                   # GCN, GAT, GIN, Multi-Task
│   ├── training/                 # Eğitim döngüsü / Training loop
│   ├── explainability/           # GNNExplainer
│   ├── interaction/              # DDI + Alternatif / Alternative
│   └── utils/                    # Yardımcı / Utilities
├── webapp/                       # Streamlit web uygulaması
├── reports/                      # Akademik raporlar
├── data/                         # Ham ve işlenmiş veri
├── checkpoints/                  # Model ağırlıkları
└── tests/                        # Birim testleri
```

---

## 📊 Sonuçlar | Results

> Sonuçlar model eğitimi tamamlandıktan sonra güncellenecektir.
> Results will be updated after model training is completed.

---

## 🤝 Katkıda Bulunanlar | Contributors

| İsim / Name | Rol / Role | Öğrenci No / Student ID |
|---|---|---|
| Hasan | Proje Lideri / Project Lead | 230212925 |

**Ders / Course:** Derin Öğrenme / Deep Learning  
**Üniversite / University:** Osim Teknik Üniversitesi / Osim Technical University  
**Dönem / Semester:** 2025-2026 Bahar / Spring

---

## 📜 Lisans | License

MIT License © 2025 Hasan — Osim Technical University

---

<p align="center">⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın! / Star if you like it! ⭐</p>
