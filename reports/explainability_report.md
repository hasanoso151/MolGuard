# 🧠 MolGuard — Açıklanabilirlik Raporu | Explainability Report

**Öğrenci / Student:** Hasan (230212925)

---

## 1. Giriş | Introduction

Yapay zeka modellerinin sağlık alanında kullanımı için açıklanabilirlik kritiktir. Bir model bir ilacın toksik olduğunu tahmin ettiğinde, "neden?" sorusuna cevap verebilmelidir.

Explainability is critical for AI in healthcare. When a model predicts a drug is toxic, it must be able to answer "why?"

---

## 2. Yöntemler | Methods

### 2.1 GNNExplainer

- Her molekül için düğüm ve kenar maskeleri optimize edilmiştir (200 epoch)
- Maske değerleri 0-1 arasında: 1 = çok önemli, 0 = önemsiz
- Optimizasyon: Maskelerin entropy'si minimize edilirken tahmin doğruluğu korunur

### 2.2 GAT Dikkat Analizi

- GAT modelinin son katmanından düğüm embedding normları çıkarılmıştır
- Yüksek norm = model o atoma daha fazla "dikkat" ediyor
- Kenar önemi: kaynak ve hedef düğüm önemlerinin ortalaması

### 2.3 Embedding Uzayı Analizi (t-SNE)

- GCN'in son katmanından 256 boyutlu graf gömmeleri çıkarılmıştır
- t-SNE ile 2 boyuta indirgenmiştir (perplexity=30)
- Toksik ve non-toksik moleküllerin uzaydaki dağılımı incelenmiştir

---

## 3. Bulgular | Findings

### 3.1 GNNExplainer Sonuçları

Analiz edilen moleküllerde tutarlı olarak:

1. **Yüksek önemli atomlar**: N (azot), O (oksijen) — fonksiyonel gruplar
2. **Orta önemli**: Aromatik halka karbon atomları
3. **Düşük önemli**: Alifatik (zincir) karbon atomları

Bu bulgular kimyasal toksikoloji literatürü ile uyumludur:
- Azot içeren gruplar (amino, nitro) reaktif metabolitler oluşturabilir
- Oksijen içeren gruplar (hidroksi, karbonil) oksidatif strese katkıda bulunabilir

### 3.2 Dikkat Haritası Sonuçları

GAT dikkat analizi, GNNExplainer sonuçlarını desteklemiştir:
- Heteroatomlar (N, O, S) tutarlı olarak yüksek dikkat skoru almıştır
- Halka bağlantı noktaları (junction atoms) ortalama üstü dikkat göstermiştir

### 3.3 Embedding Uzayı

t-SNE görselleştirmesi:
- Toksik ve non-toksik moleküller kısmen ayrışmış kümeler oluşturmuştur
- Tam ayrışma olmaması, toksisite tahmininin zorluğunu yansıtmaktadır
- Multi-Task gömmeleri, farklı veri setlerinden gelen moleküllerin anlamlı gruplamalar gösterdiğini ortaya koymuştur

---

## 4. Güvenli Alternatif Mekanizması | Safe Alternative Mechanism

Embedding uzayında kosinüs benzerliği kullanılarak:
1. Hedef molekülün gömmesi hesaplanır
2. Tüm test moleküllerinin gömmeleri ile benzerlik hesaplanır
3. Benzer (>0.8 kosinüs) AMA daha düşük toksisite skoruna sahip moleküller önerilir

Bu yöntem, kimyasal olarak benzer ama güvenli alternatifleri bulmak için pratik bir yaklaşımdır.

---

## 5. Sonuç | Conclusion

GNNExplainer ve dikkat analizi, modelin kararlarının kimyasal olarak anlamlı olduğunu göstermiştir. Bu, modelin sadece istatistiksel örüntüleri değil, gerçek kimyasal bilgiyi öğrendiğine işaret etmektedir.

---

*MolGuard © 2025*
