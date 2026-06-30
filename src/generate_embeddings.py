import os
import gc
import json
import torch
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

def main():
    print("=== COSNUP: BELLEK KORUMALI & HIZLI MULTI-FIELD E5-LARGE VEKTÖR ÜRETİMİ ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # processed klasörü yoksa oluşturalım
    os.makedirs(processed_path, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Kullanılacak Cihaz: {device.upper()}")
    if device == "cuda":
        print(f"    - Ekran Kartı: {torch.cuda.get_device_name(0)}")
        
    print("\n[2] multilingual-e5-large modeli yükleniyor (1024 boyutlu)...")
    # FP16 kullanarak VRAM ve RAM tasarrufu yapıyoruz
    model = SentenceTransformer('intfloat/multilingual-e5-large', device=device, model_kwargs={"torch_dtype": torch.float16})
    print("    - Model GPU'ya başarıyla yüklendi.")
    
    # ----------------------------------------------------
    # A. ARAMA SORGULARI (TERMS) VEKTÖRLEŞTİRME
    # ----------------------------------------------------
    print("\n[3] Arama Sorguları (Terms) işleniyor...")
    terms_df = pl.read_csv(os.path.join(raw_path, "terms.csv"))
    terms_df = terms_df.with_columns([pl.col("query").fill_null("").alias("clean_query")])
    
    term_texts = ["query: " + q for q in terms_df["clean_query"].to_list()]
    term_ids = terms_df["term_id"].to_list()
    
    print(f"    - Toplam {len(term_texts)} sorgu vektörleştiriliyor...")
    term_embeddings = model.encode(term_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    # Haritayı oluşturup hemen JSON olarak kaydedelim (Bellek tasarrufu için)
    print("    - Sorgu indeks haritası kaydediliyor...")
    term_id_to_idx = {t_id: idx for idx, t_id in enumerate(term_ids)}
    with open(os.path.join(processed_path, "term_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(term_id_to_idx, f)
    del term_id_to_idx, terms_df, term_texts
    gc.collect()

    # Sorgu vektörlerini diskte saklayıp RAM'den derhal temizliyoruz
    print("    - Sorgu vektörleri kaydediliyor ve RAM boşaltılıyor...")
    np.save(os.path.join(processed_path, "term_embeddings.npy"), term_embeddings.astype(np.float16))
    del term_embeddings
    gc.collect()
    
    # ----------------------------------------------------
    # B. ÜRÜN BİLGİLERİ (ITEMS) VEKTÖRLEŞTİRME (OPTIMIZED)
    # ----------------------------------------------------
    print("\n[4] Ürün Kataloğu (Items) yükleniyor...")
    items_df = pl.read_csv(os.path.join(raw_path, "items.csv"))
    
    items_df = items_df.with_columns([
        pl.col("title").fill_null("").alias("clean_title"),
        pl.col("category").fill_null("").alias("clean_category"),
        pl.col("attributes").fill_null("").alias("clean_attributes")
    ])
    
    item_ids = items_df["item_id"].to_list()
    
    # Ürün indeks haritasını oluşturup kaydedelim ve RAM'den temizleyelim
    print("    - Ürün indeks haritası kaydediliyor...")
    item_id_to_idx = {i_id: idx for idx, i_id in enumerate(item_ids)}
    with open(os.path.join(processed_path, "item_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(item_id_to_idx, f)
    del item_id_to_idx
    gc.collect()
    
    # 1. Başlık Temsilleri
    title_texts = ["passage: " + t for t in items_df["clean_title"].to_list()]
    print("\n    - 1/3: Ürün Başlıkları vektörleştiriliyor...")
    title_embeddings = model.encode(title_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    # Başlık vektörünü kaydet ve derhal RAM'den temizle
    print("      > Başlık vektörleri kaydediliyor ve RAM boşaltılıyor...")
    np.save(os.path.join(processed_path, "item_title_embeddings.npy"), title_embeddings.astype(np.float16))
    del title_embeddings, title_texts
    gc.collect()
    
    # 2. Kategori Temsilleri (Sadece Benzersiz Olanlar!)
    unique_categories = items_df["clean_category"].unique().to_list()
    unique_cat_texts = ["passage: " + c for c in unique_categories]
    print(f"\n    - 2/3: Kategori Hiyerarşileri vektörleştiriliyor (Toplam {len(items_df)} satırdan sadece {len(unique_categories)} adet benzersiz kategori işlenecek)...")
    unique_cat_embeddings = model.encode(unique_cat_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    # Geri haritalama (Mapping back to full size)
    print("      > Kategoriler tam boyutlu matrise haritalanıyor...")
    cat_to_emb = dict(zip(unique_categories, unique_cat_embeddings))
    cat_embeddings = np.array([cat_to_emb[c] for c in items_df["clean_category"].to_list()], dtype=np.float16)
    
    # Kategori vektörünü kaydet ve derhal RAM'den temizle
    print("      > Kategori vektörleri kaydediliyor ve RAM boşaltılıyor...")
    np.save(os.path.join(processed_path, "item_category_embeddings.npy"), cat_embeddings)
    del cat_embeddings, unique_categories, unique_cat_texts, unique_cat_embeddings, cat_to_emb
    gc.collect()
    
    # 3. Öznitelik Temsilleri (Sadece Benzersiz Olanlar!)
    unique_attributes = items_df["clean_attributes"].unique().to_list()
    unique_attr_texts = ["passage: " + a for a in unique_attributes]
    print(f"\n    - 3/3: Ürün Özellikleri vektörleştiriliyor (Toplam {len(items_df)} satırdan sadece {len(unique_attributes)} adet benzersiz özellik işlenecek)...")
    unique_attr_embeddings = model.encode(unique_attr_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    # Geri haritalama (Mapping back to full size)
    print("      > Özellikler tam boyutlu matrise haritalanıyor...")
    attr_to_emb = dict(zip(unique_attributes, unique_attr_embeddings))
    attr_embeddings = np.array([attr_to_emb[a] for a in items_df["clean_attributes"].to_list()], dtype=np.float16)
    
    # Özellik vektörünü kaydet ve derhal RAM'den temizle
    print("      > Özellik vektörleri kaydediliyor ve RAM boşaltılıyor...")
    np.save(os.path.join(processed_path, "item_attributes_embeddings.npy"), attr_embeddings)
    del attr_embeddings, unique_attributes, unique_attr_texts, unique_attr_embeddings, attr_to_emb, items_df
    gc.collect()
    
    print("\n✔ Tüm multi-field E5-large vektörleri ve haritaları diske başarıyla kaydedildi!")
    print("=== İŞLEM TAMAMLANDI ===")

if __name__ == "__main__":
    main()