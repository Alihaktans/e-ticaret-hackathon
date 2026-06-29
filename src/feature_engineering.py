import os
import gc
import polars as pl
import numpy as np
from snowballstemmer import stemmer

# Türkçe kök bulucunun tanımlanması
turk_stemmer = stemmer('turkish')

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
    """Metindeki kelimelerin Türkçe köklerini bulur"""
    if not text:
        return ""
    words = text.split()
    return " ".join(turk_stemmer.stemWords(words))

def get_char_ngrams_python(text, n=3):
    """Metni karakter düzeyinde n-gram'lara böler (Boşlukları kaldırarak)"""
    text_clean = str(text).replace(" ", "")
    if not text_clean:
        return []
    if len(text_clean) < n:
        return [text_clean]
    return [text_clean[i:i+n] for i in range(len(text_clean) - n + 1)]

def build_features_polars(pairs_df, items_df, terms_df, mode="train"):
    """
    Kök bulma ve karakter n-gram özelliklerini içeren gelişmiş pipeline.
    """
    # Birleştirme (Join)
    df = pairs_df.join(items_df, on="item_id", how="left")
    df = df.join(terms_df, on="term_id", how="left")
    
    # Kelime listelerine bölme (Orijinal ve Kök halleri için)
    df = df.with_columns([
        pl.col("clean_query").str.split(" ").alias("q_words"),
        pl.col("clean_title").str.split(" ").alias("t_words"),
        pl.col("clean_category").str.split(" ").alias("cat_words"),
        pl.col("clean_attributes").str.split(" ").alias("attr_words"),
        
        # Kök kelimelerin listeye bölünmesi
        pl.col("stem_query").str.split(" ").alias("q_stem_words"),
        pl.col("stem_title").str.split(" ").alias("t_stem_words")
    ])
    
    # Küme kesişimleri (Orijinal metinler için)
    df = df.with_columns([
        pl.col("q_words").list.set_intersection("t_words").alias("intersect_words"),
        pl.col("q_words").list.set_union("t_words").alias("union_words"),
        
        # Kök metinler için kesişimler
        pl.col("q_stem_words").list.set_intersection("t_stem_words").alias("intersect_stem_words"),
        pl.col("q_stem_words").list.set_union("t_stem_words").alias("union_stem_words"),
        
        # Karakter N-Gram kesişimleri
        pl.col("q_3gram").list.set_intersection("t_3gram").alias("intersect_3gram"),
        pl.col("q_3gram").list.set_union("t_3gram").alias("union_3gram"),
        
        pl.col("q_4gram").list.set_intersection("t_4gram").alias("intersect_4gram"),
        pl.col("q_4gram").list.set_union("t_4gram").alias("union_4gram")
    ])
    
    # Metriklerin hesaplanması
    df = df.with_columns([
        # 1. Kelime Düzeyinde Benzerlikler (Orijinal)
        (pl.col("intersect_words").list.len() / pl.col("union_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_sim"),
        (pl.col("intersect_words").list.len() / pl.col("q_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage"),
        
        # 2. Türkçe Kök Düzeyinde Benzerlikler (Yeni)
        (pl.col("intersect_stem_words").list.len() / pl.col("union_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_stemmed"),
        (pl.col("intersect_stem_words").list.len() / pl.col("q_stem_words").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_stemmed"),
        
        # 3. Karakter 3-Gram Benzerlikleri (Yeni)
        (pl.col("intersect_3gram").list.len() / pl.col("union_3gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_3gram"),
        (pl.col("intersect_3gram").list.len() / pl.col("q_3gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_3gram"),
        
        # 4. Karakter 4-Gram Benzerlikleri (Yeni)
        (pl.col("intersect_4gram").list.len() / pl.col("union_4gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("jaccard_4gram"),
        (pl.col("intersect_4gram").list.len() / pl.col("q_4gram").list.len()).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias("query_coverage_4gram"),
        
        # Diğer özellikler
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
    
    cols_to_keep = [
        'term_id', 'item_id', 
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed',
        'jaccard_3gram', 'query_coverage_3gram',
        'jaccard_4gram', 'query_coverage_4gram'
    ]
    if "id" in df.columns:
        cols_to_keep.append("id")
    if "label" in df.columns:
        cols_to_keep.append("label")
        
    return df.select(cols_to_keep)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    print("=== ADIM 5: GELİŞMİŞ TÜRKÇE NLP ÖZELLİK MÜHENDİSLİĞİ PIPELINE ===\n")
    
    print("[1] Katalog verileri yükleniyor ve temizleniyor...")
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
    
    print("   - Katalog kelime kökleri ve n-gram'lar çıkarılıyor...")
    # Katalog düzeyinde (bir kez) hızlı python map işlemleri
    items_pd = items.to_pandas()
    terms_pd = terms.to_pandas()
    
    items_pd['stem_title'] = items_pd['clean_title'].apply(stem_text_python)
    terms_pd['stem_query'] = terms_pd['clean_query'].apply(stem_text_python)
    
    items_pd['t_3gram'] = items_pd['clean_title'].apply(lambda x: get_char_ngrams_python(x, 3))
    terms_pd['q_3gram'] = terms_pd['clean_query'].apply(lambda x: get_char_ngrams_python(x, 3))
    
    items_pd['t_4gram'] = items_pd['clean_title'].apply(lambda x: get_char_ngrams_python(x, 4))
    terms_pd['q_4gram'] = terms_pd['clean_query'].apply(lambda x: get_char_ngrams_python(x, 4))
    
    items = pl.from_pandas(items_pd[['item_id', 'clean_title', 'clean_category', 'clean_attributes', 'clean_brand', 'stem_title', 't_3gram', 't_4gram']])
    terms = pl.from_pandas(terms_pd[['term_id', 'clean_query', 'stem_query', 'q_3gram', 'q_4gram']])
    
    # 2. Train İşleme (Yığınlar Halinde)
    train_pairs_path = os.path.join(processed_path, "train_with_negatives.csv")
    print(f"\n[2] Eğitim kümesi yükleniyor: {train_pairs_path}")
    train_pairs = pl.read_csv(train_pairs_path)
    
    train_out_path = os.path.join(processed_path, "train_features.csv")
    process_in_batches(train_pairs, items, terms, out_path=train_out_path, mode="train", chunk_size=500_000)
    
    del train_pairs
    gc.collect()
    
    # 3. Test İşleme (Yığınlar Halinde)
    test_pairs_path = os.path.join(raw_path, "submission_pairs.csv")
    print(f"[3] Test (submission) kümesi yükleniyor: {test_pairs_path}")
    test_pairs = pl.read_csv(test_pairs_path)
    
    test_out_path = os.path.join(processed_path, "test_features.csv")
    process_in_batches(test_pairs, items, terms, out_path=test_out_path, mode="test", chunk_size=500_000)
    
    del test_pairs
    gc.collect()
    
    print("=== TÜM GELİŞMİŞ ÖZELLİKLER BAŞARIYLA OLUŞTURULDU VE KAYDEDİLDİ ===")

def process_in_batches(pairs_df, items_df, terms_df, out_path, mode="train", chunk_size=500_000):
    total_rows = len(pairs_df)
    num_chunks = int(np.ceil(total_rows / chunk_size))
    
    print(f"   - Toplam {total_rows:,} satır, {num_chunks} parça halinde işlenecek.")
    
    if os.path.exists(out_path):
        os.remove(out_path)
        
    for i in range(0, total_rows, chunk_size):
        chunk_num = (i // chunk_size) + 1
        print(f"     > Parça {chunk_num}/{num_chunks} hesaplanıyor...")
        
        chunk = pairs_df[i : i + chunk_size]
        chunk_features = build_features_polars(chunk, items_df, terms_df, mode=mode)
        
        with open(out_path, "ab") as f:
            chunk_features.write_csv(f, include_header=(i == 0))
            
        del chunk, chunk_features
        gc.collect()
        
    print(f"   ✔ {mode.upper()} kümesi diske kaydedildi: {out_path}\n")

if __name__ == "__main__":
    main()