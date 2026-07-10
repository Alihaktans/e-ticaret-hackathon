# Trendyol E-Commerce Relevance

Trendyol E-Ticaret Hackathonu için arama sorgusu–ürün alaka tahmini deneyleri.
Proje; lexical özellikler, embedding modelleri, gradient boosting, cross-encoder
ve aday yeniden sıralama yaklaşımlarını içerir.

## Proje durumu

Bu dal bir araştırma/yarışma çalışma alanıdır. `scripts/` altındaki numaralı dosyalar
kronolojik deney geçmişidir; her script üretim pipeline'ının zorunlu parçası değildir.
Yeni çalışma yaparken önce [deney rehberine](scripts/README.md) bakın.

## Kurulum

Python 3.10 veya 3.11 önerilir.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml,dev]"
```

GPU tabanlı PyTorch kurulumu CUDA sürümüne göre ayrıca yapılmalıdır. PyTorch'u
kurduktan sonra `.[nlp]` seçeneğini kullanabilirsiniz.

## Veri yerleşimi

Kaggle verileri repoya commit edilmez. Dosyaları aşağıdaki şekilde yerleştirin:

```text
data/
  raw/
    items.csv
    terms.csv
    training_pairs.csv
    submission_pairs.csv
    sample_submission.csv
  processed/
models/
reports/
submissions/
```

Veri kurulumunu kontrol etmek için:

```bash
python main.py audit
```

## Temel akış

1. `scripts/00_audit_data.py` — ham veri şeması ve kimlik kontrolleri.
2. `scripts/01_build_training_dataset.py` — eğitim çiftleri ve negatif örnekler.
3. `scripts/04_build_embedding_feature_cache.py` — yeniden kullanılabilir embedding cache'i.
4. Sonraki numaralı scriptler — bağımsız deney ve submission varyantları.

Bir deneyi repo kökünden çalıştırın:

```bash
python scripts/00_audit_data.py
```

## Paket yapısı

- `src/trendyol/config.py`: ortak dizinler ve beklenen veri şeması
- `src/trendyol/data_io.py`: güvenli CSV okuma ve veri doğrulama
- `src/trendyol/text_preprocess.py`: Türkçe metin normalizasyonu
- `src/trendyol/embedding.py`: embedding model/cache yardımcıları
- `src/trendyol/negative_sampling.py`: sentetik negatif üretimi
- `src/trendyol/metrics.py`: Macro-F1 ve threshold yardımcıları
- `scripts/`: kronolojik deney geçmişi

## Yarışma güvenliği

- `data/`, modeller, cache'ler, raporlar ve submission dosyaları Git'e eklenmez.
- Public leaderboard sonucuna göre aşırı threshold/budget seçimi yapmayın.
- Query veya item kimliği aynı olan satırları fold'lar arasında bölmeden önce leakage
  riskini kontrol edin.
- Final adayını yalnızca leaderboard ile değil, group-based OOF Macro-F1 ile seçin.
