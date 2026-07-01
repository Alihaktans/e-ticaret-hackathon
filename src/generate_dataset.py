import os
import gc
import json
import random
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text_simple(text):
    """Metin temizleme"""
    if pd.isna(text):
        return ""
    text = str(text).replace('İ', 'i').replace('I', 'ı').lower()
    return text

def run_dataset_generation(neg_ratio=6, hard_neg_pct=0.80):
    print("=== ADIM 4: TF-IDF TABANLI GELİŞMİŞ HARD NEGATIVE MINING ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    os.makedirs(processed_path, exist_ok=True)
    
    items_path = os.path.join(raw_path, "items.csv")
    train_path = os.path.join(raw_path, "training_pairs.csv")
    terms_path = os.path.join(raw_path, "terms.csv")
    
    print("[1] Dosyalar yükleniyor...")
    items = pd.read_csv(items_path, usecols=["item_id", "title", "category"])
    train_pairs = pl.read_csv(train_path)
    terms = pd.read_csv(terms_path)
    
    print("   - Metinler hızlıca temizleniyor...")
    items['clean_title'] = items['title'].apply(clean_text_simple)
    terms['clean_query'] = terms['query'].apply(clean_text_simple)
    
    # 1:6 oranına göre adet hesaplamaları
    num_hard_negs = int(neg_ratio * hard_neg_pct) # 5 adet
    num_easy_negs = neg_ratio - num_hard_negs      # 1 adet
    
    print(f"✔ Dosyalar başarıyla yüklendi. (Zor Negatif Oranı: {num_hard_negs}, Kolay Negatif Oranı: {num_easy_negs})")

    # 2. Hızlı Erişim İndeksleri
    print("\n[2] Hızlı arama indeksleri hazırlanıyor...")
    items['cat_l1'] = items['category'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else 'NONE')
    item_to_cat = dict(zip(items['item_id'], items['cat_l1']))
    cat_to_items = items.groupby('cat_l1')['item_id'].apply(list).to_dict()
    all_items = list(items['item_id'].unique())
    
    # Hızlı lookup için ID-Index eşleşmeleri
    term_ids = terms['term_id'].to_list()
    term_id_to_idx = {tid: idx for idx, tid in enumerate(term_ids)}
    
    item_ids = items['item_id'].to_list()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    # Pozitif çiftleri gruplayalım: term_id -> pozitif ürün setleri
    from collections import defaultdict
    pos_pairs_set = set(zip(train_pairs['term_id'], train_pairs['item_id']))
    positives = defaultdict(set)
    for row in train_pairs.iter_rows(named=True):
        positives[row['term_id']].add(row['item_id'])

    # 3. TF-IDF Vektörleştirme
    print("\n[3] Katalog TF-IDF uzayına aktarılıyor...")
    vectorizer = TfidfVectorizer(max_features=25000, lowercase=False)
    X_items = vectorizer.fit_transform(items['clean_title'])
    X_queries = vectorizer.transform(terms['clean_query'])
    
    # 4. Gelişmiş Hard Negative Arama Algoritması (Batch-by-Batch)
    print("\n[4] Matematiksel en yakın zor negatifler aranıyor (Bu işlem sadece 1-2 dakika sürecektir)...")
    unique_train_terms = train_pairs['term_id'].unique().to_list()
    num_unique_terms = len(unique_train_terms)
    
    term_to_hard_negs = {}
    batch_size = 100  # Bellek kullanımını 770 MB'a düşürmek için 100'erli yığınlar
    random.seed(42)
    
    for start_idx in tqdm(range(0, num_unique_terms, batch_size), desc="Zor Negatif Arama"):
        end_idx = min(start_idx + batch_size, num_unique_terms)
        batch_terms = unique_train_terms[start_idx:end_idx]
        
        # Sorgu TF-IDF vektörleri
        batch_term_indices = [term_id_to_idx[tid] for tid in batch_terms]
        q_batch = X_queries[batch_term_indices]
        
        # Katalog ile hızlı matris çarpımı (Kosinüs Benzerliği Matrisi)
        # float32 kullanarak bellek kullanımını bir kez daha %50 oranında düşürüyoruz
        sim_matrix = q_batch.dot(X_items.T).toarray().astype(np.float32)
        
        for idx, term_id in enumerate(batch_terms):
            scores = sim_matrix[idx]
            
            # Pozitif ürünlerin benzerlik skorlarını -1 yapıyoruz ki negatif olarak seçilmesinler
            pos_items = positives[term_id]
            pos_indices = [item_id_to_idx[iid] for iid in pos_items if iid in item_id_to_idx]
            scores[pos_indices] = -1.0
            
            # En yüksek benzerliğe sahip ilk 15 aday ürünü buluyoruz (Çeşitlilik için argpartition)
            top_k_indices = np.argpartition(scores, -15)[-15:]
            
            # Kelime bazlı benzerliği 0'dan büyük olanları seçiyoruz
            candidates = [item_ids[i] for i in top_k_indices if scores[i] > 0.0]
            
            # Eğer kelime uyuşması olan yeterli aday yoksa, kategori içi rastgele ürünlere dön (Safe Fallback)
            if len(candidates) < num_hard_negs:
                pos_item_id = list(pos_items)[0]
                pos_item_cat = item_to_cat.get(pos_item_id, 'NONE')
                cat_pool = cat_to_items.get(pos_item_cat, all_items)
                candidates = candidates + [random.choice(cat_pool) for _ in range(15 - len(candidates))]
                
            term_to_hard_negs[term_id] = candidates
            
        del sim_matrix
        gc.collect()

    # 5. Yeni Veri Setinin Birleştirilmesi
    print("\n[5] Yeni zor negatif verileri birleştiriliyor...")
    neg_records = []
    
    for row in tqdm(train_pairs.iter_rows(named=True), total=len(train_pairs), desc="Eğitim Seti Oluşturuluyor"):
        term_id = row['term_id']
        pos_item_id = row['item_id']
        
        # 5a. 1 Adet Kolay Negatif (Rastgele)
        easy_sampled = 0
        while easy_sampled < num_easy_negs:
            random_item = random.choice(all_items)
            if (term_id, random_item) not in pos_pairs_set:
                neg_records.append({
                    'term_id': term_id,
                    'item_id': random_item,
                    'label': 0
                })
                easy_sampled += 1
                
        # 5b. 5 Adet TF-IDF Zor Negatif (Top-15 aday arasından rastgele çeşitlendirilmiş)
        candidates = term_to_hard_negs.get(term_id, all_items)
        sampled_hard = random.sample(candidates, num_hard_negs)
        for item in sampled_hard:
            neg_records.append({
                'term_id': term_id,
                'item_id': item,
                'label': 0
            })

    # Verileri DataFrame'e çevirip kaydetme
    neg_df = pd.DataFrame(neg_records)
    train_pairs_df = train_pairs.to_pandas()
    train_pairs_df['label'] = 1
    
    final_df = pd.concat([train_pairs_df[['term_id', 'item_id', 'label']], neg_df], ignore_index=True)
    # Karıştırma
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = os.path.join(processed_path, "train_with_negatives.csv")
    print(f"   - Dosya kaydediliyor: {output_file}")
    final_df.to_csv(output_file, index=False)
    
    print("\n✔ Yeni Gelişmiş Eğitim Seti Dağılımı:")
    print(final_df['label'].value_counts())
    print("\n=== ADIM 4 BAŞARIYLA TAMAMLANDI ===")

if __name__ == "__main__":
    run_dataset_generation()