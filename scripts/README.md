# Deney scriptleri

Bu klasör kronolojik yarışma günlüğüdür. Script numarası yükseldikçe daha yeni bir
deney anlamına gelir; ancak daha yeni olması otomatik olarak daha iyi olduğu anlamına
gelmez. Dosyalar geriye dönük karşılaştırmalar için korunur.

## Aşamalar

| Aralık | Ana amaç |
|---|---|
| `00–14` | veri denetimi, ilk embedding modelleri ve temel karşılaştırmalar |
| `15–31` | hard-negative üretimi, semantic ranker ve threshold araması |
| `32–64` | PU learning, class-prior tahmini ve manuel hata analizi |
| `65–93` | cross-encoder, BGE/MiniLM ve query-level kalibrasyon |
| `94–121` | CatBoost feature tabloları, voting ve sparse sinyaller |
| `122–164` | gelişmiş lexical/semantic guard ve pairwise modeller |
| `165–193` | Qwen denetimleri, Trendyol embedding ve contrastive eğitim |
| `194–217` | bağımsız sınıflandırıcılar, rulepack ve final audit çalışmaları |

## Çalıştırma kuralları

- Komutları repo kökünden çalıştırın: `python scripts/<dosya>.py`.
- Scriptin başındaki giriş/çıkış sabitlerini çalıştırmadan önce kontrol edin.
- Yeni script, önceki artefaktı kullanıyorsa gerekli dosyayı ve üreten scripti dosya
  başındaki docstring'de belirtin.
- Deney çıktısını `data/processed/`, `reports/`, `models/` veya `submissions/`
  altında tutun; bu dizinler Git'e eklenmez.
- Final skor kararını yalnızca public leaderboard'a göre vermeyin. Group-based OOF
  Macro-F1, class-wise F1 ve değişen satır sayısını birlikte kaydedin.

## Yeni deney şablonu

Yeni numarayı mevcut en yüksek numaradan sonra verin ve dosya adını eylemle başlayan
açıklayıcı bir isim yapın:

```text
218_train_oof_meta_ranker.py
219_make_calibrated_candidates.py
```

Birbirinin küçük varyasyonu olan çok sayıda script yerine parametreli tek script ve
CSV/JSON deney özeti tercih edin.

