import os
import pandas as pd
import random
from tqdm import tqdm

def run_dataset_generation(neg_ratio=6, hard_neg_pct=0.80):
    print("=== ADIM 4: HIZLI AKILLI NEGATİF ÖRNEKLEME VE VERİ SETİ OLUŞTURMA ===\n")
    
    # 1. Dosya Yolları
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    # processed klasörü yoksa oluşturalım
    os.makedirs(processed_path, exist_ok=True)
    
    items_path = os.path.join(raw_path, "items.csv")
    train_path = os.path.join(raw_path, "training_pairs.csv")
    
    print("[1] Dosyalar yükleniyor...")
    items = pd.read_csv(items_path, usecols=["item_id", "category"])
    train_pairs = pd.read_csv(train_path, usecols=["term_id", "item_id"])
    print(f"✔ {len(items)} ürün ve {len(train_pairs)} pozitif eğitim çifti yüklendi.\n")
    
    # 2. Kategori Ayrıştırma
    print("[2] Ürün ana kategorileri ayrıştırılıyor...")
    items['cat_l1'] = items['category'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else 'NONE')
    
    # Hızlı erişim için sözlükler oluşturuyoruz
    item_to_cat = dict(zip(items['item_id'], items['cat_l1']))
    # All_items'ı NumPy array yerine çok daha hızlı çalışan Python listesine çeviriyoruz
    all_items = list(items['item_id'].unique())
    
    print("   - Kategori indeksleri oluşturuluyor...")
    cat_to_items = items.groupby('cat_l1')['item_id'].apply(list).to_dict()
    
    # Pozitif çiftleri küme (set) haline getirme
    pos_pairs_set = set(zip(train_pairs['term_id'], train_pairs['item_id']))
    
    # 3. Negatif Örnekleme Algoritması
    print("\n[3] Hızlı Negatif Örnekleme Başlatılıyor (Python random.choice ile)...")
    neg_records = []
    
    num_hard_negs = int(neg_ratio * hard_neg_pct)
    num_easy_negs = neg_ratio - num_hard_negs
    
    # Hızlı ve tekrarlanabilir sonuçlar için seed ayarlıyoruz
    random.seed(42)
    
    for _, row in tqdm(train_pairs.iterrows(), total=len(train_pairs), desc="Örnekleniyor"):
        term_id = row['term_id']
        pos_item_id = row['item_id']
        
        pos_item_cat = item_to_cat.get(pos_item_id, 'NONE')
        
        # 3a. Kolay Negatifler
        easy_sampled = 0
        while easy_sampled < num_easy_negs:
            random_item = random.choice(all_items) # NumPy yerine standart random
            if (term_id, random_item) not in pos_pairs_set:
                neg_records.append({
                    'term_id': term_id,
                    'item_id': random_item,
                    'label': 0
                })
                easy_sampled += 1
                
        # 3b. Zor Negatifler
        hard_sampled = 0
        cat_items_pool = cat_to_items.get(pos_item_cat, all_items)
        if len(cat_items_pool) < 10:
            cat_items_pool = all_items
            
        while hard_sampled < num_hard_negs:
            random_item = random.choice(cat_items_pool) # NumPy yerine standart random
            if (term_id, random_item) not in pos_pairs_set:
                neg_records.append({
                    'term_id': term_id,
                    'item_id': random_item,
                    'label': 0
                })
                hard_sampled += 1

    # 4. Veri Setlerini Birleştirme ve Kaydetme
    print("\n[4] Veri kümeleri birleştiriliyor...")
    neg_df = pd.DataFrame(neg_records)
    
    train_pairs['label'] = 1
    final_df = pd.concat([train_pairs, neg_df], ignore_index=True)
    
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = os.path.join(processed_path, "train_with_negatives.csv")
    print(f"   - Dosya kaydediliyor: {output_file}")
    final_df.to_csv(output_file, index=False)
    
    print("\n✔ Yeni Eğitim Seti Dağılımı:")
    print(final_df['label'].value_counts())
    print("\n=== ADIM 4 BAŞARIYLA TAMAMLANDI ===")

if __name__ == "__main__":
    run_dataset_generation(neg_ratio=6, hard_neg_pct=0.80)