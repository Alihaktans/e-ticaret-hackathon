import os
import gc
import json
import polars as pl
import numpy as np
from snowballstemmer import stemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Türkçe kök bulucunun tanımlanması
turk_stemmer = stemmer('turkish')
global_vectorizer = None

def clean_text_polars(col_name):
    return (
        pl.col(col_name)
        .fill_null("")
        .cast(pl.String)
        .str.replace_all("İ", "i")
        .str.replace_all("I", "ı")
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def stem_text_python(text):
    if not text:
        return ""
    words = text.split()
    stemmed_words = []
    for word in words:
        try:
            stemmed_words.append(turk_stemmer.stemWord(word))
        except Exception:
            stemmed_words.append(word)
    return " ".join(stemmed_words)

def build_features_polars(pairs_df, items_df, terms_df, term_map, item_map, term_emb, item_emb, mode="train"):
    # 1. Birleştirme
    df = pairs_df.join(items_df, on="item_id", how="left")
    df = df.join(terms_df, on="term_id", how="left")
    
    # 2. Kelime listelerine bölme ve küme kesişimleri
    df = df.with_columns([
        pl.col("clean_query").str.split(" ").alias("q_words"),
        pl.col("clean_title").str.split(" ").alias("t_words"),
        pl.col("clean_category").str.split(" ").alias("cat_words"),
        pl.col("clean_attributes").str.split(" ").alias("attr_words"),
        pl.col("stem_query").str.split(" ").alias("q_stem_words"),
        pl.col("stem_title").str.split(" ").alias("t_stem_words")
    ])
    
    df = df.with_columns([
        pl.col("q_words").list.set_intersection("t_words").alias("intersect_words"),
        pl.col("q_words").list.set_union("t_words").alias("union_words"),
        pl.col("q_stem_words").list.set_intersection("t_stem_words").alias("intersect_stem_words"),
        pl.col("q_stem_words").list.set_union("t_stem_words").alias("union_stem_words")
    ])
    
    # 3. Klasik Benzerlikler
    df = df.with_columns([
        (pl.col("intersect_words").list.len() / pl.col("union_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_sim"),
        (pl.col("intersect_words").list.len() / pl.col("q_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage"),
        (pl.col("intersect_stem_words").list.len() / pl.col("union_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_stemmed"),
        (pl.col("intersect_stem_words").list.len() / pl.col("q_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_stemmed"),
        
        pl.when((pl.col("color_attr") != "") & (pl.col("clean_query").str.contains(pl.col("color_attr"), literal=True))).then(1.0).otherwise(0.0).cast(pl.Float32).alias("color_match"),
        pl.when((pl.col("material_attr") != "") & (pl.col("clean_query").str.contains(pl.col("material_attr"), literal=True))).then(1.0).otherwise(0.0).cast(pl.Float32).alias("material_match"),
        
        pl.col("clean_title").str.contains(pl.col("clean_query"), literal=True).cast(pl.Float32).alias("exact_match"),
        pl.when((pl.col("clean_brand") != "") & (pl.col("clean_brand") != "missing_value") & (pl.col("clean_query").str.contains(pl.col("clean_brand"), literal=True))).then(1.0).otherwise(0.0).cast(pl.Float32).alias("brand_in_query"),
        (pl.col("q_words").list.set_intersection("cat_words").list.len() / pl.col("q_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("cat_overlap"),
        (pl.col("q_words").list.set_intersection("attr_words").list.len() / pl.col("q_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("attr_overlap"),
        
        pl.col("q_words").list.len().cast(pl.Int8).alias("query_word_len"),
        pl.col("t_words").list.len().cast(pl.Int8).alias("title_word_len")
    ])
    
    df = df.with_columns([
        (pl.col("title_word_len") - pl.col("query_word_len")).cast(pl.Int8).alias("len_diff")
    ])
    
    # 4. TF-IDF Cosine Benzerliği
    queries_clean = df['clean_query'].to_list()
    titles_clean = df['clean_title'].to_list()
    q_tfidf = global_vectorizer.transform(queries_clean)
    t_tfidf = global_vectorizer.transform(titles_clean)
    tfidf_sim = np.array(q_tfidf.multiply(t_tfidf).sum(axis=1)).ravel()
    
    # 5. BERT Semantic Benzerliği (Yeni!)
    term_ids = df['term_id'].to_list()
    item_ids = df['item_id'].to_list()
    
    # Haritadan indeksleri buluyoruz (Bulamazsa güvenli olarak 0 döner)
    q_indices = [term_map.get(tid, 0) for tid in term_ids]
    t_indices = [item_map.get(iid, 0) for iid in item_ids]
    
    # Vektörleri matristen çekip doğrudan çarpıyoruz (L2 normalize oldukları için bu Kosinüs benzerliğidir)
    q_vecs = term_emb[q_indices]
    t_vecs = item_emb[t_indices]
    bert_sim = np.sum(q_vecs * t_vecs, axis=1)
    
    df = df.with_columns([
        pl.Series("tfidf_sim", tfidf_sim).cast(pl.Float32),
        pl.Series("bert_sim", bert_sim).cast(pl.Float32)
    ])
    
    cols_to_keep = [
        'term_id', 'item_id', 
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed', 
        'tfidf_sim', 'color_match', 'material_match', 'bert_sim' # bert_sim eklendi
    ]
    if "id" in df.columns:
        cols_to_keep.append("id")
    if "label" in df.columns:
        cols_to_keep.append("label")
        
    return df.select(cols_to_keep)

def main():
    global global_vectorizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    print("=== ADIM 5: HİBRİT (TF-IDF + BERT) ÖZELLİK MÜHENDİSLİĞİ PIPELINE ===\n")
    
    # BERT Haritalarını ve Matrislerini Belleğe Yükleme
    print("[1] BERT Embedding matrisleri ve haritaları yükleniyor...")
    with open(os.path.join(processed_path, "term_mapping.json"), "r") as f:
        term_map = json.load(f)
    with open(os.path.join(processed_path, "item_mapping.json"), "r") as f:
        item_map = json.load(f)
        
    term_emb = np.load(os.path.join(processed_path, "term_embeddings.npy"))
    item_emb = np.load(os.path.join(processed_path, "item_embeddings.npy"))
    print("    ✔ BERT Vektörleri başarıyla belleğe alındı.")
    
    print("\n[2] Katalog verileri yükleniyor ve temizleniyor...")
    items = pl.read_csv(os.path.join(raw_path, "items.csv"))
    terms = pl.read_csv(os.path.join(raw_path, "terms.csv"))
    
    items = items.with_columns([
        clean_text_polars("title").alias("clean_title"),
        clean_text_polars("category").alias("clean_category"),
        clean_text_polars("attributes").alias("clean_attributes"),
        clean_text_polars("brand").alias("clean_brand")
    ])
    
    terms = terms.with_columns([
        clean_text_polars("query").alias("clean_query")
    ])
    
    items = items.with_columns([
        pl.col("clean_attributes").str.extract(r"renk\s*:\s*([^,]+)", 1).fill_null("").str.strip_chars().alias("color_attr"),
        pl.col("clean_attributes").str.extract(r"materyal\s*:\s*([^,]+)", 1).fill_null("").str.strip_chars().alias("material_attr")
    ])
    
    terms = terms.with_columns([
        pl.col("clean_query").map_elements(stem_text_python, return_dtype=pl.String).alias("stem_query")
    ])
    items = items.with_columns([
        pl.col("clean_title").map_elements(stem_text_python, return_dtype=pl.String).alias("stem_title")
    ])
    
    print("    - TF-IDF Vektörleştirici eğitiliyor...")
    all_titles = items.select("clean_title").to_numpy().ravel().tolist()
    global_vectorizer = TfidfVectorizer(max_features=50000, lowercase=False)
    global_vectorizer.fit(all_titles)
    
    items = items.select(['item_id', 'clean_title', 'clean_category', 'clean_attributes', 'clean_brand', 'stem_title', 'color_attr', 'material_attr'])
    terms = terms.select(['term_id', 'clean_query', 'stem_query'])
    
    # 3. Train İşleme
    train_pairs_path = os.path.join(processed_path, "train_with_negatives.csv")
    print(f"\n[3] Eğitim kümesi yükleniyor: {train_pairs_path}")
    train_pairs = pl.read_csv(train_pairs_path)
    
    train_out_path = os.path.join(processed_path, "train_features.csv")
    process_in_batches(train_pairs, items, terms, term_map, item_map, term_emb, item_emb, out_path=train_out_path, mode="train", chunk_size=500_000)
    
    del train_pairs
    gc.collect()
    
    # 4. Test İşleme
    test_pairs_path = os.path.join(raw_path, "submission_pairs.csv")
    print(f"[4] Test (submission) kümesi yükleniyor: {test_pairs_path}")
    test_pairs = pl.read_csv(test_pairs_path)
    
    test_out_path = os.path.join(processed_path, "test_features.csv")
    process_in_batches(test_pairs, items, terms, term_map, item_map, term_emb, item_emb, out_path=test_out_path, mode="test", chunk_size=500_000)
    
    del test_pairs
    gc.collect()
    
    print("=== HİBRİT ÖZELLİKLER BAŞARIYLA OLUŞTURULDU VE KAYDEDİLDİ ===")

def process_in_batches(pairs_df, items_df, terms_df, term_map, item_map, term_emb, item_emb, out_path, mode="train", chunk_size=500_000):
    total_rows = len(pairs_df)
    num_chunks = int(np.ceil(total_rows / chunk_size))
    print(f"    - Toplam {total_rows:,} satır, {num_chunks} parça halinde işlenecek.")
    
    if os.path.exists(out_path):
        os.remove(out_path)
        
    for i in range(0, total_rows, chunk_size):
        chunk_num = (i // chunk_size) + 1
        print(f"      > Parça {chunk_num}/{num_chunks} hesaplanıyor...")
        
        chunk = pairs_df[i : i + chunk_size]
        chunk_features = build_features_polars(chunk, items_df, terms_df, term_map, item_map, term_emb, item_emb, mode=mode)
        
        with open(out_path, "ab") as f:
            chunk_features.write_csv(f, include_header=(i == 0))
            
        del chunk, chunk_features
        gc.collect()
        
    print(f"    ✔ {mode.upper()} kümesi diske kaydedildi: {out_path}\n")

if __name__ == "__main__":
    main()