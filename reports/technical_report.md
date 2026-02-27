# 🧬 MolGuard — Teknik Rapor | Technical Report

**Proje:** Çok Görevli Grafik Sinir Ağları ile İlaç Güvenliği Tahmin Sistemi
**Project:** Multi-Task Graph Neural Network for Drug Safety Prediction

**Öğrenci / Student:** Hasan (230212925)
**Ders / Course:** Derin Öğrenme / Deep Learning
**Üniversite / University:** Osim Teknik Üniversitesi
**Tarih / Date:** 2025-2026 Bahar / Spring

---

## 1. Giriş | Introduction

İlaç güvenliği, sağlık sektörünün en kritik konularından biridir. Dünya Sağlık Örgütü'ne göre, advers ilaç reaksiyonları (ADR) dünya genelinde hastaneye yatışların %5-8'inden sorumludur. Bu proje, Grafik Sinir Ağları (GNN) kullanarak ilaç moleküllerinin güvenlik profillerini tahmin eden kapsamlı bir yapay zeka sistemi geliştirmektedir.

Drug safety is one of the most critical issues in healthcare. According to the WHO, adverse drug reactions (ADR) are responsible for 5-8% of hospital admissions worldwide. This project develops a comprehensive AI system that predicts drug safety profiles using Graph Neural Networks (GNN).

### 1.1 Motivasyon | Motivation

- İlaç geliştirme sürecinde güvenlik testleri uzun ve maliyetlidir
- Drug safety testing is lengthy and expensive
- Mevcut yöntemler tüm olası yan etkileri öngöremez
- Current methods cannot predict all possible side effects
- Yapay zeka tabanlı yaklaşımlar bu süreci hızlandırabilir
- AI-based approaches can accelerate this process

### 1.2 Katkılar | Contributions

1. **Çok Görevli GNN Mimarisi**: Toksisite, yan etkiler, etkinlik ve klinik güvenliği birlikte öğrenen paylaşılan omurga modeli / Multi-Task GNN with shared backbone learning toxicity, side effects, efficacy, and clinical safety jointly
2. **3 GNN Karşılaştırması**: GCN, GAT ve GIN mimarilerinin sistematik karşılaştırması / Systematic comparison of GCN, GAT, and GIN architectures
3. **Açıklanabilir Tahminler**: GNNExplainer ile hangi atomların tahmine katkıda bulunduğunun görselleştirilmesi / Explainable predictions showing which atoms contribute using GNNExplainer
4. **Hasta Bazlı Risk**: Hasta profiline göre kişiselleştirilmiş risk değerlendirmesi / Personalized risk assessment based on patient profile
5. **Güvenli Alternatif Önerici**: Embedding uzayında benzer ama daha güvenli molekül önerisi / Safer molecule suggestion using embedding space similarity

---

## 2. İlgili Çalışmalar | Related Work

### 2.1 Moleküler Grafik Sinir Ağları | Molecular GNNs

Moleküller doğal olarak grafik yapıları olarak temsil edilebilir: atomlar düğümleri, kimyasal bağlar ise kenarları oluşturur. Bu temsil, GNN'lerin moleküler özellikleri öğrenmesi için ideal bir temel sağlar.

Molecules can naturally be represented as graph structures: atoms form nodes, chemical bonds form edges. This representation provides an ideal basis for GNNs to learn molecular properties.

- **GCN (Kipf & Welling, 2017)**: Spektral tabanlı graf evrişimi / Spectral-based graph convolution
- **GAT (Veličković et al., 2018)**: Dikkat mekanizmalı graf ağları / Graph networks with attention mechanism
- **GIN (Xu et al., 2019)**: Weisfeiler-Leman testi kadar güçlü ayırt edici / As powerful as WL test in distinguishing graphs

### 2.2 Çok Görevli Öğrenme | Multi-Task Learning

Çok görevli öğrenme, birden fazla ilişkili görevi aynı anda öğrenerek genelleme performansını artırır (Caruana, 1997). İlaç güvenliğinde toksisite, yan etkiler ve etkinlik birbiriyle ilişkili görevlerdir.

Multi-task learning improves generalization by learning multiple related tasks simultaneously. In drug safety, toxicity, side effects, and efficacy are interrelated tasks.

### 2.3 Açıklanabilir Yapay Zeka | Explainable AI

GNNExplainer (Ying et al., 2019), graf düzeyinde tahminler için düğüm ve kenar maskelerini optimize ederek açıklamalar üretir. Bu, ilaç güvenliğinde hangi moleküler alt yapıların riskli olduğunu anlamak için kritiktir.

GNNExplainer generates explanations by optimizing node and edge masks for graph-level predictions, critical for understanding which molecular substructures are risky.

---

## 3. Yöntem | Methodology

### 3.1 Veri Setleri | Datasets

| Veri Seti | Görev | Molekül Sayısı | Görev Sayısı | Kaynak |
|---|---|---|---|---|
| Tox21 | Toksisite tahmini | 7,831 | 12 | MoleculeNet/OGB |
| SIDER | Yan etki tahmini | 1,427 | 27 | MoleculeNet/OGB |
| BBBP | Kan-beyin bariyeri | 2,039 | 1 | MoleculeNet/OGB |
| ClinTox | Klinik toksisite | 1,478 | 2 | MoleculeNet/OGB |

Tüm veri setleri OGB (Open Graph Benchmark) üzerinden scaffold split ile bölünmüştür. Scaffold split, kimyasal yapısal çeşitliliği koruyarak daha gerçekçi bir değerlendirme sağlar.

All datasets are split using scaffold split from OGB, which preserves chemical structural diversity for more realistic evaluation.

### 3.2 Moleküler Graf Temsili | Molecular Graph Representation

Her molekül bir grafik G = (V, E) olarak temsil edilir:
- **Düğümler (V)**: Atomlar — 9 boyutlu özellik vektörü (atom numarası, kiralite, derece, biçimsel yük, hidrojen sayısı, radikal elektron, hibridizasyon, aromatiklik, halka üyeliği)
- **Kenarlar (E)**: Kimyasal bağlar — 3 boyutlu özellik vektörü (bağ tipi, stereo yapılandırma, konjugasyon)

### 3.3 Model Mimarileri | Model Architectures

#### 3.3.1 GCN (Graph Convolutional Network)
- 4 GCN katmanı, 256 gizli boyut
- BatchNorm + ReLU + Dropout (0.3) her katmanda
- Global ortalama havuzlama
- 2 katmanlı MLP tahmin kafası

#### 3.3.2 GAT (Graph Attention Network)
- 4 GAT katmanı, 8 dikkat kafası
- 256 gizli boyut (32 × 8 kafa)
- Çoklu kafalar arası birleştirme

#### 3.3.3 GIN (Graph Isomorphism Network)
- 5 GIN katmanı, 256 gizli boyut
- 2 katmanlı MLP her GIN katmanında
- Global toplama havuzlama (add pooling)

#### 3.3.4 Multi-Task GNN
- Paylaşılan GAT omurgası (4 katman, 8 kafa)
- 4 görev-özel tahmin kafası (Tox21: 12, SIDER: 27, BBBP: 1, ClinTox: 2)
- Round-robin eğitim stratejisi

### 3.4 Eğitim Detayları | Training Details

| Parametre | Değer |
|---|---|
| Optimizer | Adam |
| Öğrenme oranı / Learning rate | 0.001 |
| Ağırlık bozunumu / Weight decay | 1e-5 |
| Zamanlayıcı / Scheduler | Cosine Annealing (min_lr=1e-5) |
| Batch boyutu / Batch size | 64 |
| Maksimum epoch | 100 (tek görev) / 80 (çok görev) |
| Erken durdurma sabrı / Patience | 15 epoch |
| Kayıp fonksiyonu / Loss | Binary Cross-Entropy with Logits |
| Değerlendirme metriği / Metric | ROC-AUC |

---

## 4. Sonuçlar | Results

### 4.1 Tek Görevli Model Karşılaştırması (Tox21) | Single-Task Comparison

| Model | Valid AUC | Test AUC | En İyi Epoch | Süre (s) |
|---|---|---|---|---|
| **GCN** | **0.7883** | **0.7479** | 84 | 130 |
| GAT | 0.7711 | 0.7315 | 30 | 76 |
| GIN | 0.7171 | 0.6819 | 14 | 40 |

**Analiz**: GCN, Tox21 veri setinde en iyi performansı göstermiştir. GAT'ın dikkat mekanizması ek karmaşıklık getirirken belirgin bir avantaj sağlamamıştır. GIN'in düşük performansı, veri setinin küçük boyutu nedeniyle aşırı güçlü modelin genelleme yapamamasından kaynaklanabilir.

### 4.2 Çok Görevli Model Sonuçları | Multi-Task Results

| Görev / Task | Veri Seti | Test AUC |
|---|---|---|
| Toksisite / Toxicity | Tox21 | 0.7431 |
| Yan Etkiler / Side Effects | SIDER | 0.6415 |
| Etkinlik / Efficacy | BBBP | 0.6883 |
| Klinik Güvenlik / Clinical | ClinTox | **0.8100** |

**Analiz**: Multi-Task model, ClinTox'ta en yüksek performansı (0.81) göstermiştir. Toksisite görevinde tek görevli GCN'e yakın performans elde etmiştir (0.7431 vs 0.7479), bu da paylaşılan omurganın farklı görevlerden bilgi transfer edebildiğini göstermektedir.

### 4.3 Açıklanabilirlik Sonuçları | Explainability Results

GNNExplainer analizi, toksisiteye katkıda bulunan atomların genellikle aşağıdaki özelliklere sahip olduğunu ortaya koymuştur:
- Azot (N) ve oksijen (O) içeren fonksiyonel gruplar yüksek önem skoru
- Aromatik halka yapıları orta düzeyde önem
- Karbon (C) zinciri düşük önem

### 4.4 Hasta Risk Simülasyonu | Patient Risk Simulation

Hasta profili koşullandırması, aynı ilacın farklı hasta gruplarında farklı risk seviyeleri gösterdiğini doğrulamıştır:
- Genç sağlıklı bireyler: Düşük risk (×0.7 çarpan)
- Yaşlı + böbrek hastalığı: Yüksek risk (×1.8 çarpan)

---

## 5. Tartışma | Discussion

### 5.1 Güçlü Yönler | Strengths

1. **Kapsamlı sistem**: Tek bir mimari ile 4 farklı güvenlik boyutunu değerlendirme
2. **Açıklanabilirlik**: Kara kutu olmayan, yorumlanabilir tahminler
3. **Kişiselleştirme**: Hasta bazlı risk değerlendirmesi
4. **Pratik uygulama**: Web arayüzü ile erişilebilir sistem

### 5.2 Sınırlılıklar | Limitations

1. Hasta koşullandırma katmanı basit çarpan modeli kullanmaktadır; gerçek klinik verilerle eğitilmemiştir
2. SIDER veri setindeki düşük AUC (0.64), 27 görevin bazılarında yetersiz veri olduğuna işaret etmektedir
3. İlaç-ilaç etkileşimi modeli henüz tam entegre edilmemiştir

### 5.3 Gelecek Çalışmalar | Future Work

1. Gerçek hasta verileriyle koşullandırma katmanının eğitilmesi
2. DRKG (Drug Repurposing Knowledge Graph) entegrasyonu
3. 3D moleküler yapı bilgisinin dahil edilmesi
4. Klinik doğrulama çalışmaları

---

## 6. Sonuç | Conclusion

MolGuard, Grafik Sinir Ağları'nın ilaç güvenliği tahmininde etkin bir şekilde kullanılabileceğini göstermiştir. Çok görevli öğrenme yaklaşımı, özellikle ClinTox görevinde 0.81 AUC ile güçlü performans sergilemiştir. GNNExplainer ile sağlanan açıklanabilirlik, modelin güvenilirliğini artırmaktadır.

MolGuard demonstrates that Graph Neural Networks can be effectively used for drug safety prediction. The multi-task approach showed strong performance especially on ClinTox (0.81 AUC). Explainability through GNNExplainer enhances model trustworthiness.

---

## 7. Kaynaklar | References

1. Kipf, T.N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
2. Veličković, P. et al. (2018). Graph Attention Networks. ICLR.
3. Xu, K. et al. (2019). How Powerful are Graph Neural Networks? ICLR.
4. Ying, R. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS.
5. Hu, W. et al. (2020). Open Graph Benchmark. NeurIPS.
6. Caruana, R. (1997). Multitask Learning. Machine Learning.
7. Wu, Z. et al. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science.

---

*MolGuard © 2025 — Osim Teknik Üniversitesi / Osim Technical University*
