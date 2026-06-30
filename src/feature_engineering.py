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
    """Polars Expressions kullanarak Rust seviyesinde çok hızlı metin temizleme"""
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
    """Metindeki kelimelerin Türkçe köklerini bulur (Tamamen Hata Korumalı)"""
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

def get_char_ngrams_python(text, n=3):
    """Metni karakter düzeyinde n-gram'lara böler (Boşlukları kaldırarak)"""
    text_clean = str(text).replace(" ", "")
    if not text_clean:
        return []
    if len(text_clean) < n:
        return [text_clean]
    return [text_clean[i:i+n] for i in range(len(text_clean) - n + 1)]

def build_features_polars(pairs_df, items_df, terms_df, term_map, item_map, term_emb, title_emb, cat_emb, attr_emb, mode="train"):
    """
    On-the-fly n-gram hesaplaması içeren bellek dostu pipeline.
    """
    df = pairs_df.join(items_df, on="item_id", how="left")
    df = df.join(terms_df, on="term_id", how="left")
    
    # 1. Bellek Tasarrufu İçin N-Gram'ları Sadece Bu Aktif Parça (500k) İçin Üretiyoruz
    df = df.with_columns([
        pl.col("clean_query").map_elements(lambda x: get_char_ngrams_python(x, 3), return_dtype=pl.List(pl.String)).alias("q_3gram"),
        pl.col("clean_title").map_elements(lambda x: get_char_ngrams_python(x, 3), return_dtype=pl.List(pl.String)).alias("t_3gram"),
        pl.col("clean_query").map_elements(lambda x: get_char_ngrams_python(x, 4), return_dtype=pl.List(pl.String)).alias("q_4gram"),
        pl.col("clean_title").map_elements(lambda x: get_char_ngrams_python(x, 4), return_dtype=pl.List(pl.String)).alias("t_4gram")
    ])
    
    # Kelime listelerine bölme
    df = df.with_columns([
        pl.col("clean_query").str.split(" ").alias("q_words"),
        pl.col("clean_title").str.split(" ").alias("t_words"),
        pl.col("clean_category").str.split(" ").alias("cat_words"),
        pl.col("clean_attributes").str.split(" ").alias("attr_words"),
        pl.col("stem_query").str.split(" ").alias("q_stem_words"),
        pl.col("stem_title").str.split(" ").alias("t_stem_words")
    ])
    
    # Küme kesişimleri
    df = df.with_columns([
        pl.col("q_words").list.set_intersection("t_words").alias("intersect_words"),
        pl.col("q_words").list.set_union("t_words").alias("union_words"),
        
        pl.col("q_stem_words").list.set_intersection("t_stem_words").alias("intersect_stem_words"),
        pl.col("q_stem_words").list.set_union("t_stem_words").alias("union_stem_words"),
        
        pl.col("q_3gram").list.set_intersection("t_3gram").alias("intersect_3gram"),
        pl.col("q_3gram").list.set_union("t_3gram").alias("union_3gram"),
        
        pl.col("q_4gram").list.set_intersection("t_4gram").alias("intersect_4gram"),
        pl.col("q_4gram").list.set_union("t_4gram").alias("union_4gram")
    ])
    
    # Benzerliklerin hesaplanması
    df = df.with_columns([
        (pl.col("intersect_words").list.len() / pl.col("union_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_sim"),
        (pl.col("intersect_words").list.len() / pl.col("q_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage"),
        
        (pl.col("intersect_stem_words").list.len() / pl.col("union_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_stemmed"),
        (pl.col("intersect_stem_words").list.len() / pl.col("q_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_stemmed"),
        
        (pl.col("intersect_3gram").list.len() / pl.col("union_3gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_3gram"),
        (pl.col("intersect_3gram").list.len() / pl.col("q_3gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_3gram"),
        
        (pl.col("intersect_4gram").list.len() / pl.col("union_4gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_4gram"),
        (pl.col("intersect_4gram").list.len() / pl.col("q_4gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_4gram"),
        
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
    
    # TF-IDF Cosine Benzerliği
    print("     > TF-IDF Kosinüs Benzerliği hesaplanıyor...")
    queries_clean = df['clean_query'].to_list()
    titles_clean = df['clean_title'].to_list()
    q_tfidf = global_vectorizer.transform(queries_clean)
    t_tfidf = global_vectorizer.transform(titles_clean)
    tfidf_sim = np.array(q_tfidf.multiply(t_tfidf).sum(axis=1)).ravel()
    
    # Çoklu BERT (E5-Large) Semantik Benzerlikleri
    term_ids = df['term_id'].to_list()
    item_ids = df['item_id'].to_list()
    
    q_indices = [term_map.get(tid, 0) for tid in term_ids]
    t_indices = [item_map.get(iid, 0) for iid in item_ids]
    
    # SSD'den sadece gerekli satırlar anlık olarak okunur (Hafıza dostudur)
    q_vecs = term_emb[q_indices]
    
    bert_sim_title = np.sum(q_vecs * title_emb[t_indices], axis=1)
    bert_sim_category = np.sum(q_vecs * cat_emb[t_indices], axis=1)
    bert_sim_attributes = np.sum(q_vecs * attr_emb[t_indices], axis=1)
    
    df = df.with_columns([
        pl.Series("tfidf_sim", tfidf_sim).cast(pl.Float32),
        pl.Series("bert_sim_title", bert_sim_title).cast(pl.Float32),
        pl.Series("bert_sim_category", bert_sim_category).cast(pl.Float32),
        pl.Series("bert_sim_attributes", bert_sim_attributes).cast(pl.Float32)
    ])
    
    cols_to_keep = [
        'term_id', 'item_id', 
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed', 'tfidf_sim',
        'color_match', 'material_match',
        'bert_sim_title', 'bert_sim_category', 'bert_sim_attributes',
        'jaccard_3gram', 'query_coverage_3gram',
        'jaccard_4gram', 'query_coverage_4gram'
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
    
    print("=== ADIM 5: ŞAMPİYON ÇOKLU ALAN SEMANTİK PIPELINE ===\n")
    
    # Vektörleri ve haritaları belleğe alıyoruz (Bellek Haritalamalı mmap_mode='r')
    print("[1] BERT (E5-Large) Çoklu Alan matrisleri diskten haritalanıyor (mmap)...")
    with open(os.path.join(processed_path, "term_mapping.json"), "r") as f:
        term_map = json.load(f)
    with open(os.path.join(processed_path, "item_mapping.json"), "r") as f:
        item_map = json.load(f)
        
    # mmap_mode='r' sayesinde bu dosyalar RAM'e yüklenmez, doğrudan diskten (SSD) on-the-fly okunur.
    term_emb = np.load(os.path.join(processed_path, "term_embeddings.npy"), mmap_mode='r')
    title_emb = np.load(os.path.join(processed_path, "item_title_embeddings.npy"), mmap_mode='r')
    cat_emb = np.load(os.path.join(processed_path, "item_category_embeddings.npy"), mmap_mode='r')
    attr_emb = np.load(os.path.join(processed_path, "item_attributes_embeddings.npy"), mmap_mode='r')
    print("    ✔ 4 ayrı BERT matrisi diske başarıyla haritalandı (Bellek kullanımı: ~0 MB).")
    
    print("\n[2] Katalog verileri Polars ile yükleniyor...")
    items = pl.read_csv(os.path.join(raw_path, "items.csv"))
    terms = pl.read_csv(os.path.join(raw_path, "terms.csv"))
    
    items = items.with_columns([
        clean_text_polars("title").alias("clean_title"),
        clean_text_polars("category").alias("clean_category"),
        clean_text_polars("attributes").alias("clean_attributes"),
        clean_text_polars("brand").alias("clean_brand")
    ])
    terms = terms.with_columns([clean_text_polars("query").alias("clean_query")])
    
    print("   - Katalog özniteliklerinden Renk ve Materyal bilgileri ayıklanıyor...")
    items = items.with_columns([
        pl.col("clean_attributes").str.extract(r"renk\s*:\s*([^,]+)", 1).fill_null("").str.strip_chars().alias("color_attr"),
        pl.col("clean_attributes").str.extract(r"materyal\s*:\s*([^,]+)", 1).fill_null("").str.strip_chars().alias("material_attr")
    ])
    
    print("   - Katalog kelime kökleri çıkarılıyor (Saf Polars)...")
    terms = terms.with_columns([
        pl.col("clean_query").map_elements(stem_text_python, return_dtype=pl.String).alias("stem_query")
    ])
    items = items.with_columns([
        pl.col("clean_title").map_elements(stem_text_python, return_dtype=pl.String).alias("stem_title")
    ])
    
    items = items.select(['item_id', 'clean_title', 'clean_category', 'clean_attributes', 'clean_brand', 'stem_title', 'color_attr', 'material_attr'])
    terms = terms.select(['term_id', 'clean_query', 'stem_query'])
    
    print("   - TF-IDF Vektörleştirici eğitiliyor (Katalog üzerinden)...")
    all_titles = items.select("clean_title").to_numpy().ravel().tolist()
    global_vectorizer = TfidfVectorizer(max_features=50000, lowercase=False)
    global_vectorizer.fit(all_titles)
    
    # 3. Train İşleme
    train_pairs_path = os.path.join(processed_path, "train_with_negatives.csv")
    print(f"\n[3] Eğitim kümesi yükleniyor: {train_pairs_path}")
    train_pairs = pl.read_csv(train_pairs_path)
    
    train_out_path = os.path.join(processed_path, "train_features.csv")
    process_in_batches(train_pairs, items, terms, term_map, item_map, term_emb, title_emb, cat_emb, attr_emb, out_path=train_out_path, mode="train", chunk_size=500_000)
    
    del train_pairs
    gc.collect()
    
    # 4. Test İşleme
    test_pairs_path = os.path.join(raw_path, "submission_pairs.csv")
    print(f"[4] Test (submission) kümesi yükleniyor: {test_pairs_path}")
    test_pairs = pl.read_csv(test_pairs_path)
    
    test_out_path = os.path.join(processed_path, "test_features.csv")
    process_in_batches(test_pairs, items, terms, term_map, item_map, term_emb, title_emb, cat_emb, attr_emb, out_path=test_out_path, mode="test", chunk_size=500_000)
    
    del test_pairs
    gc.collect()
    
    print("=== TÜM GELİŞMİŞ ÖZELLİKLER BAŞARIYLA OLUŞTURULDU VE KAYDEDİLDİ ===")

def process_in_batches(pairs_df, items_df, terms_df, term_map, item_map, term_emb, title_emb, cat_emb, attr_emb, out_path, mode="train", chunk_size=500_000):
    total_rows = len(pairs_df)
    num_chunks = int(np.ceil(total_rows / chunk_size))
    print(f"    - Toplam {total_rows:,} satır, {num_chunks} parça halinde işlenecek.")
    
    if os.path.exists(out_path):
        os.remove(out_path)
        
    for i in range(0, total_rows, chunk_size):
        chunk_num = (i // chunk_size) + 1
        print(f"      > Parça {chunk_num}/{num_chunks} hesaplanıyor...")
        
        chunk = pairs_df[i : i + chunk_size]
        chunk_features = build_features_polars(chunk, items_df, terms_df, term_map, item_map, term_emb, title_emb, cat_emb, attr_emb, mode=mode)
        
        with open(out_path, "ab") as f:
            chunk_features.write_csv(f, include_header=(i == 0))
            
        del chunk, chunk_features
        gc.collect()
        
    print(f"    ✔ {mode.upper()} kümesi diske kaydedildi: {out_path}\n")

if __name__ == "__main__":
    main()