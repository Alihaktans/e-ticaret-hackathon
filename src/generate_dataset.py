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

def generate_split_dataset(pairs_df, unique_terms, positives, term_id_to_idx, item_id_to_idx, item_ids, 
                           X_queries, X_items, term_emb, title_emb, item_to_cat, cat_to_items, 
                           all_items, pos_pairs_set, num_hard_lex=3, num_hard_sem=2, num_easy=1, batch_size=250):
    """
    Belirli bir alt küme (Train veya Val) için kelimesel ve anlamsal zor negatifleri üretir.
    float32 ve çoklu çekirdek AVX desteğiyle en yüksek hızda çalışır.
    """
    term_to_hard_negs = {}
    num_terms = len(unique_terms)
    
    for start_idx in range(0, num_terms, batch_size):
        end_idx = min(start_idx + batch_size, num_terms)
        batch_terms = unique_terms[start_idx:end_idx]
        
        batch_term_indices = [term_id_to_idx[tid] for tid in batch_terms]
        
        # 1a. Kelimesel (Lexical/BM25) Benzerlik Arama
        q_batch_tfidf = X_queries[batch_term_indices]
        sim_lexical = q_batch_tfidf.dot(X_items.T).toarray().astype(np.float32) # float32 ile CPU AVX hızı
        
        # 1b. Anlamsal (Semantic/E5-Large) Benzerlik Arama (float32 ile çoklu çekirdek desteği)
        q_batch_emb = term_emb[batch_term_indices]
        sim_semantic = np.dot(q_batch_emb, title_emb.T)
        
        for idx, term_id in enumerate(batch_terms):
            pos_items = positives[term_id]
            pos_indices = [item_id_to_idx[iid] for iid in pos_items if iid in item_id_to_idx]
            
            # Pozitif ürünleri eliyoruz
            sim_lexical[idx, pos_indices] = -1.0
            sim_semantic[idx, pos_indices] = -1.0
            
            # En yakın 15 kelimesel ve 15 anlamsal aday ürünü seçiyoruz
            top_lex_indices = np.argpartition(sim_lexical[idx], -15)[-15:]
            top_sem_indices = np.argpartition(sim_semantic[idx], -15)[-15:]
            
            lex_candidates = [item_ids[i] for i in top_lex_indices if sim_lexical[idx, i] > 0.0]
            sem_candidates = [item_ids[i] for i in top_sem_indices if sim_semantic[idx, i] > 0.0]
            
            # Safe Fallback (Eğer yeterli eşleşme yoksa kategori içi rastgele seç)
            pos_item_id = list(pos_items)[0]
            pos_item_cat = item_to_cat.get(pos_item_id, 'NONE')
            cat_pool = cat_to_items.get(pos_item_cat, all_items)
            
            if len(lex_candidates) < num_hard_lex:
                lex_candidates = lex_candidates + [random.choice(cat_pool) for _ in range(15 - len(lex_candidates))]
            if len(sem_candidates) < num_hard_sem:
                sem_candidates = sem_candidates + [random.choice(cat_pool) for _ in range(15 - len(sem_candidates))]
                
            term_to_hard_negs[term_id] = {
                "lexical": lex_candidates,
                "semantic": sem_candidates
            }
            
        del sim_lexical, sim_semantic
        gc.collect()
        
    # Negatif kayıtları inşa etme
    neg_records = []
    for row in pairs_df.iter_rows(named=True):
        term_id = row['term_id']
        pos_item_id = row['item_id']
        
        # 1 Adet Kolay Negatif (Rastgele)
        easy_sampled = 0
        while easy_sampled < num_easy:
            random_item = random.choice(all_items)
            if (term_id, random_item) not in pos_pairs_set:
                neg_records.append({
                    'term_id': term_id,
                    'item_id': random_item,
                    'label': 0
                })
                easy_sampled += 1
                
        # 3 Adet Kelimesel Zor Negatif
        candidates_lex = term_to_hard_negs[term_id]["lexical"]
        sampled_lex = random.sample(candidates_lex, num_hard_lex)
        for item in sampled_lex:
            neg_records.append({'term_id': term_id, 'item_id': item, 'label': 0})
            
        # 2 Adet Anlamsal Zor Negatif
        candidates_sem = term_to_hard_negs[term_id]["semantic"]
        sampled_sem = random.sample(candidates_sem, num_hard_sem)
        for item in sampled_sem:
            neg_records.append({'term_id': term_id, 'item_id': item, 'label': 0})
            
    # Pozitif ve Negatifleri birleştirme
    neg_df = pd.DataFrame(neg_records)
    pos_df = pairs_df.to_pandas()
    pos_df['label'] = 1
    
    final_df = pd.concat([pos_df[['term_id', 'item_id', 'label']], neg_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df

def main():
    print("=== ADIM 1: SIZINTISIZ SOTA VERİ KÜMESİ ÜRETİMİ ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # 1. Dosyaların Yüklenmesi
    print("[1] Dosyalar yükleniyor...")
    items = pd.read_csv(os.path.join(raw_path, "items.csv"), usecols=["item_id", "title", "category"])
    train_pairs = pl.read_csv(os.path.join(raw_path, "training_pairs.csv"))
    terms = pd.read_csv(os.path.join(raw_path, "terms.csv"))
    
    # İndeks haritalarını yüklüyoruz
    print("   - BERT indeks haritaları yükleniyor...")
    with open(os.path.join(processed_path, "term_mapping.json"), "r") as f:
        term_map = json.load(f)
    with open(os.path.join(processed_path, "item_mapping.json"), "r") as f:
        item_map = json.load(f)
        
    # Vektörleri doğrudan RAM'e float32 olarak yüklüyoruz (İşlemci AVX hızı için bu zorunludur!)
    print("   - BERT (E5-Large) matrisleri RAM'e yükleniyor (Float32)...")
    term_emb = np.load(os.path.join(processed_path, "term_embeddings.npy")).astype(np.float32)
    title_emb = np.load(os.path.join(processed_path, "item_title_embeddings.npy")).astype(np.float32)
    
    # Hızlı metin temizleme
    items['clean_title'] = items['title'].apply(clean_text_simple)
    terms['clean_query'] = terms['query'].apply(clean_text_simple)
    
    # 2. Hızlı Arama Sözlükleri
    print("\n[2] Hızlı arama indeksleri hazırlanıyor...")
    items['cat_l1'] = items['category'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else 'NONE')
    item_to_cat = dict(zip(items['item_id'], items['cat_l1']))
    cat_to_items = items.groupby('cat_l1')['item_id'].apply(list).to_dict()
    all_items = list(items['item_id'].unique())
    
    term_ids = terms['term_id'].to_list()
    term_id_to_idx = {tid: idx for idx, tid in enumerate(term_ids)}
    
    item_ids = items['item_id'].to_list()
    item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    
    pos_pairs_set = set(zip(train_pairs['term_id'], train_pairs['item_id']))
    from collections import defaultdict
    positives = defaultdict(set)
    for row in train_pairs.iter_rows(named=True):
        positives[row['term_id']].add(row['item_id'])

    # 3. BM25'i Taklit Eden Sublinear TF-IDF Vektörleştirme
    print("\n[3] Katalog TF-IDF (Sublinear TF) uzayına aktarılıyor...")
    vectorizer = TfidfVectorizer(max_features=30000, sublinear_tf=True, lowercase=False)
    X_items = vectorizer.fit_transform(items['clean_title'])
    X_queries = vectorizer.transform(terms['clean_query'])

    # 4. %100 SIZINTISIZ GRUP BÖLME
    print("\n[4] Sorgu bazlı sızıntısız grup bölme yapılıyor (term_id bazlı)...")
    unique_terms_all = train_pairs["term_id"].unique().sample(fraction=1.0, seed=42)
    split_idx = int(len(unique_terms_all) * 0.9)
    
    train_terms_list = unique_terms_all[0:split_idx].to_list()
    val_terms_list = unique_terms_all[split_idx:].to_list()
    
    train_pos_pairs = train_pairs.filter(pl.col("term_id").is_in(train_terms_list))
    val_pos_pairs = train_pairs.filter(pl.col("term_id").is_in(val_terms_list))
    
    print(f"    - Sızıntısız Eğitim Seti Pozitif Çift: {len(train_pos_pairs):,} | Sorgu: {len(train_terms_list):,}")
    print(f"    - Sızıntısız Doğrulama Seti Pozitif Çift: {len(val_pos_pairs):,} | Sorgu: {len(val_terms_list):,}")

    # 5. Dual Hard Negative Mining ve Dataset İnşası
    # Eğitim Seti İnşası
    print("\n[5] Sızıntısız EĞİTİM veri seti için hibrid zor negatifler aranıyor...")
    train_dataset = generate_split_dataset(
        train_pos_pairs, train_terms_list, positives, term_id_to_idx, item_id_to_idx, item_ids,
        X_queries, X_items, term_emb, title_emb, item_to_cat, cat_to_items, all_items, pos_pairs_set,
        batch_size=250 # 250'li yığınlar ile çoklu çekirdek (multi-threaded) paralelleştirmesini tetikliyoruz
    )
    train_out_path = os.path.join(processed_path, "train_pairs_split.csv")
    train_dataset.to_csv(train_out_path, index=False)
    print(f"    ✔ Eğitim veri seti başarıyla diske yazıldı: {train_out_path}")
    print(train_dataset['label'].value_counts())
    
    del train_dataset
    gc.collect()

    # Doğrulama Seti İnşası
    print("\n[6] Sızıntısız DOĞRULAMA veri seti için hibrid zor negatifler aranıyor...")
    val_dataset = generate_split_dataset(
        val_pos_pairs, val_terms_list, positives, term_id_to_idx, item_id_to_idx, item_ids,
        X_queries, X_items, term_emb, title_emb, item_to_cat, cat_to_items, all_items, pos_pairs_set,
        batch_size=250 # 250'li yığınlar ile çoklu çekirdek (multi-threaded) paralelleştirmesini tetikliyoruz
    )
    val_out_path = os.path.join(processed_path, "val_pairs_split.csv")
    val_dataset.to_csv(val_out_path, index=False)
    print(f"    ✔ Doğrulama veri seti başarıyla diske yazıldı: {val_out_path}")
    print(val_dataset['label'].value_counts())
    
    print("\n=== SIZINTISIZ SOTA VERİ KÜMELERİ BAŞARIYLA TAMAMLANDI ===")

if __name__ == "__main__":
    main()