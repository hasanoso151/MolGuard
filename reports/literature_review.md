# 📚 MolGuard — Literatür Taraması | Literature Review

**Öğrenci / Student:** Hasan (230212925)

---

## 1. Grafik Sinir Ağları | Graph Neural Networks

### 1.1 GCN — Graph Convolutional Network (Kipf & Welling, 2017)
Spektral graf teorisinden ilham alan yarı-denetimli sınıflandırma yaklaşımı. Komşu düğümlerin özelliklerini normalize ederek toplar. Basitliği ve etkinliği nedeniyle temel model olarak yaygın kullanılır.

### 1.2 GAT — Graph Attention Network (Veličković et al., 2018)
Dikkat mekanizmasını graf ağlarına uygular. Her komşunun katkısı öğrenilen dikkat katsayıları ile ağırlıklandırılır. Çoklu dikkat kafaları kullanarak farklı ilişki kalıplarını yakalar.

### 1.3 GIN — Graph Isomorphism Network (Xu et al., 2019)
Weisfeiler-Leman izomorfizm testinin ayırt etme gücüne ulaşan ilk GNN. Toplama ve MLP kombinasyonu ile en güçlü graf seviyesi temsil gücünü sağlar.

## 2. Moleküler Özellik Tahmini | Molecular Property Prediction

### 2.1 MoleculeNet (Wu et al., 2018)
Moleküler makine öğrenimi için kapsamlı kıyaslama platformu. Fizikokimyasal, biyofizik, fizyoloji ve toksisite veri setleri içerir.

### 2.2 OGB — Open Graph Benchmark (Hu et al., 2020)
Gerçekçi ve büyük ölçekli graf veri setleri koleksiyonu. Scaffold split ile kimyasal çeşitliliği koruyan standart veri bölme yöntemi sağlar.

## 3. Çok Görevli Öğrenme | Multi-Task Learning

### 3.1 Temel Kavram (Caruana, 1997)
İlişkili görevleri birlikte öğrenmek, tek tek öğrenmeye göre genelleme performansını artırır. Paylaşılan temsiller görevler arası bilgi transferini sağlar.

### 3.2 İlaç Keşfinde MTL
Moleküler özellik tahmininde çok görevli öğrenme, özellikle küçük veri setlerinde tek görevli modellere göre avantaj sağlamaktadır (Ramsundar et al., 2015).

## 4. Açıklanabilir Yapay Zeka | Explainable AI

### 4.1 GNNExplainer (Ying et al., 2019)
Graf seviyesi tahminler için düğüm ve kenar maskelerini optimize eden açıklama yöntemi. Karşılıklı bilgiyi (mutual information) maksimize ederek en bilgilendirici alt grafı bulur.

## 5. İlaç Güvenliği ve AI | Drug Safety and AI

İlaç güvenliği tahmini için derin öğrenme yaklaşımları son yıllarda büyük ilerleme kaydetmiştir. Özellikle GNN tabanlı modeller, moleküler yapıyı doğal grafik temsili ile yakalayarak geleneksel parmak izi yöntemlerine göre üstünlük göstermiştir.

---

*MolGuard © 2025*
