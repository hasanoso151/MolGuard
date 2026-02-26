# 📝 Git Commit Kuralları | Git Commit Conventions

## Format
```
<type>: <açıklama (TR)> / <description (EN)>
```

## Tipler | Types
| Tip | Açıklama / Description |
|---|---|
| `feat` | Yeni özellik / New feature |
| `fix` | Hata düzeltme / Bug fix |
| `docs` | Dokümantasyon / Documentation |
| `data` | Veri / Data related |
| `model` | Model değişiklikleri / Model changes |
| `exp` | Deney sonuçları / Experiment results |
| `viz` | Görselleştirme / Visualization |
| `webapp` | Web uygulaması / Web application |
| `report` | Rapor / Report |

## Örnekler | Examples
```bash
git commit -m "data: Tox21 veri seti yüklendi / Tox21 dataset loaded"
git commit -m "model: GCN mimarisi eklendi / GCN architecture added"
git commit -m "exp: GAT vs GCN sonuçları / GAT vs GCN results"
```
