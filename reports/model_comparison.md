# 📊 MolGuard — Model Karşılaştırma Raporu | Model Comparison Report

**Öğrenci / Student:** Hasan (230212925)

---

## 1. Karşılaştırılan Modeller | Compared Models

### GCN (Graph Convolutional Network)
- **Parametre sayısı:** ~400K
- **Yaklaşım:** Spektral tabanlı graf evrişimi
- **Havuzlama:** Global ortalama
- **Katman:** 4

### GAT (Graph Attention Network)
- **Parametre sayısı:** ~420K
- **Yaklaşım:** Dikkat mekanizmalı mesaj geçişi
- **Kafa sayısı:** 8
- **Katman:** 4

### GIN (Graph Isomorphism Network)
- **Parametre sayısı:** ~850K
- **Yaklaşım:** WL-test eşdeğeri ayırt etme gücü
- **Havuzlama:** Global toplama
- **Katman:** 5

### Multi-Task GNN
- **Parametre sayısı:** ~600K
- **Yaklaşım:** Paylaşılan GAT omurgası + görev-özel kafalar
- **Görev sayısı:** 4 (42 toplam alt görev)

---

## 2. Performans Tablosu | Performance Table

| Model | Tox21 AUC | SIDER AUC | BBBP AUC | ClinTox AUC | Toplam Süre |
|---|---|---|---|---|---|
| GCN | **0.7479** | — | — | — | 130s |
| GAT | 0.7315 | — | — | — | 76s |
| GIN | 0.6819 | — | — | — | 40s |
| Multi-Task | 0.7431 | 0.6415 | 0.6883 | **0.8100** | ~900s |

---

## 3. Analiz | Analysis

### GCN Neden En İyisi? | Why GCN Performed Best?

1. **Basitlik avantajı**: Küçük veri setlerinde basit modeller daha iyi genelleyebilir
2. **Kararlı eğitim**: GCN'in eğitim eğrisi en kararlı olandır
3. **Düzenleme**: BatchNorm + Dropout kombinasyonu GCN'de en etkili

### GAT'ın Durumu | GAT Performance

- Dikkat mekanizması ek karmaşıklık getirmiştir
- Tox21'in görece küçük boyutu dikkat mekanizmasının tam potansiyelini kullanmasını engellemiştir
- Daha büyük veri setlerinde GAT'ın üstün olması beklenir

### GIN'in Düşük Performansı | GIN Low Performance

- 5 katman + güçlü MLP aşırı öğrenmeye (overfitting) neden olmuştur
- Erken durdurma 14. epoch'ta tetiklenmiştir
- Add pooling, küçük moleküllerde mean pooling kadar etkili olmayabilir

### Multi-Task Avantajı | Multi-Task Advantage

- ClinTox'ta en yüksek performans (0.81): Transfer öğrenme etkisi
- Toksisite ve klinik güvenlik arasındaki bilgi paylaşımı faydalı
- SIDER'da düşük AUC: 27 görev çok fazla, bazılarında yetersiz veri

---

## 4. Sonuç | Conclusion

Bu karşılaştırma, model seçiminin veri seti boyutuna ve görevin karmaşıklığına bağlı olduğunu göstermektedir. Küçük moleküler veri setlerinde GCN gibi basit modeller etkilidir, ancak çok görevli öğrenme birden fazla güvenlik boyutunu tek bir modelde birleştirerek pratik değer sunmaktadır.

---

*MolGuard © 2025*
