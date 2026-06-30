import os
import json
import torch
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

def main():
    print("=== BERT (E5-BASE) VEKTÖR ÜRETİM SÜRECİ BAŞLIYOR ===\n")
    
    # 1. Dosya Yolları
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # 2. CUDA (GPU) Kontrolü
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Kullanılacak Cihaz: {device.upper()}")
    if device == "cuda":
        print(f"    - Ekran Kartı: {torch.cuda.get_device_name(0)}")
        
    # 3. Modelin Yüklenmesi
    print("\n[2] multilingual-e5-base modeli yükleniyor (İlk seferde indirebilir)...")
    # RTX 4070 için fp16 (yarı hassasiyet) kullanarak VRAM tasarrufu yapıyoruz
    model = SentenceTransformer('intfloat/multilingual-e5-base', device=device, model_kwargs={"torch_dtype": torch.float16})
    print("    - Model GPU'ya başarıyla yüklendi.")
    
    # 4. Sorguların (Terms) Vektörleştirilmesi
    print("\n[3] Arama Sorguları (Terms) işleniyor...")
    terms_df = pl.read_csv(os.path.join(raw_path, "terms.csv"))
    
    # E5 modeli için 'query: ' ön ekini ekliyoruz ve boş değerleri temizliyoruz
    terms_df = terms_df.with_columns([
        pl.col("query").fill_null("").alias("clean_query")
    ])
    term_texts = ["query: " + q for q in terms_df["clean_query"].to_list()]
    term_ids = terms_df["term_id"].to_list()
    
    print(f"    - Toplam {len(term_texts)} sorgu vektörleştiriliyor...")
    # normalize_embeddings=True sayesinde vektör boyları 1 birim yapılır (Kosinüs benzerliği için)
    term_embeddings = model.encode(
        term_texts, 
        batch_size=512, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    # 5. Ürün Başlıklarının (Items) Vektörleştirilmesi
    print("\n[4] Ürün Başlıkları (Items) işleniyor...")
    items_df = pl.read_csv(os.path.join(raw_path, "items.csv"))
    
    # E5 modeli için 'passage: ' ön ekini ekliyoruz
    items_df = items_df.with_columns([
        pl.col("title").fill_null("").alias("clean_title")
    ])
    item_texts = ["passage: " + t for t in items_df["clean_title"].to_list()]
    item_ids = items_df["item_id"].to_list()
    
    print(f"    - Toplam {len(item_texts)} ürün başlığı vektörleştiriliyor (Bu işlem birkaç dakika sürebilir)...")
    item_embeddings = model.encode(
        item_texts, 
        batch_size=512, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    # 6. ID-to-Index Haritalarının Oluşturulması
    print("\n[5] İndeks haritaları (Mapping) oluşturuluyor ve diske kaydediliyor...")
    
    term_id_to_idx = {t_id: idx for idx, t_id in enumerate(term_ids)}
    item_id_to_idx = {i_id: idx for idx, i_id in enumerate(item_ids)}
    
    # Haritaları JSON olarak kaydet
    with open(os.path.join(processed_path, "term_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(term_id_to_idx, f)
        
    with open(os.path.join(processed_path, "item_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(item_id_to_idx, f)
        
    # Vektör matrislerini Float16 (Yarı hassasiyet, çok hızlı ve %50 daha az boyut) olarak kaydet
    np.save(os.path.join(processed_path, "term_embeddings.npy"), term_embeddings.astype(np.float16))
    np.save(os.path.join(processed_path, "item_embeddings.npy"), item_embeddings.astype(np.float16))
    
    print("\n✔ Tüm vektörler ve haritalar data/processed/ klasörüne başarıyla kaydedildi!")
    print("=== İŞLEM TAMAMLANDI ===")

if __name__ == "__main__":
    main()