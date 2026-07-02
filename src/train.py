import os
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

def min_max_scale(arr):
    """Sıralama ve sınıflandırma modellerinin ham skorlarını [0, 1] arasına ölçekler"""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min + 1e-9)

def find_best_threshold(y_true, y_pred_probs):
    """En yüksek F1 skorunu veren eşik değerini bulur"""
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        preds = (y_pred_probs >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1

def main():
    print("=== ADIM 6: HİBRİT SIRALAMA & SINIFLANDIRMA ENSEMBLE EĞİTİMİ ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    processed_path = os.path.join(project_root, "data", "processed")
    
    # 1. Verilerin Yüklenmesi
    print("[1] Eğitim ve Doğrulama özellikleri yükleniyor...")
    train_df = pl.read_csv(os.path.join(processed_path, "train_features.csv"))
    val_df = pl.read_csv(os.path.join(processed_path, "val_features.csv"))
    
    # 🚨 GPU CUDA LİMİTLERİNE UYUM İÇİN GRUP SINIRLANDIRMASI (TRUNCATION)
    # Her bir arama sorgusu (term_id) için maksimum satır sayısını 1000 ile sınırlıyoruz.
    # Pozitifleri (label=1) korumak için önce label'a göre azalan sırada sıralıyoruz.
    print("   - GPU CUDA sınırlarına uyum için devasa sorgu grupları kırpılıyor...")
    
    train_df = train_df.sort(["term_id", "label"], descending=[False, True])
    train_df = train_df.with_columns(
        pl.int_range(0, pl.len()).over("term_id").alias("group_row_num")
    )
    train_df = train_df.filter(pl.col("group_row_num") < 1000).drop("group_row_num")
    
    val_df = val_df.sort(["term_id", "label"], descending=[False, True])
    val_df = val_df.with_columns(
        pl.int_range(0, pl.len()).over("term_id").alias("group_row_num")
    )
    val_df = val_df.filter(pl.col("group_row_num") < 1000).drop("group_row_num")
    
    # Sıralama modeli için nihai sıralama
    train_df = train_df.sort("term_id")
    val_df = val_df.sort("term_id")
    
    print(f"✔ Optimize Sınırsız Eğitim satırı  : {len(train_df):,}")
    print(f"✔ Optimize Sınırsız Doğrulama satırı: {len(val_df):,}")
    
    feature_cols = [
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed', 'bm25_sim',
        'color_match', 'material_match',
        'bert_sim_title', 'bert_sim_category', 'bert_sim_attributes',
        'bert_cross_sim',
        'jaccard_3gram', 'query_coverage_3gram',
        'jaccard_4gram', 'query_coverage_4gram'
    ]
    
    # CatBoost için term_id'leri benzersiz tam sayı gruplarına (group_id) çeviriyoruz
    train_df = train_df.with_columns(pl.col("term_id").cast(pl.Categorical).to_physical().alias("group_id"))
    val_df = val_df.with_columns(pl.col("term_id").cast(pl.Categorical).to_physical().alias("group_id"))
    
    X_train_all = train_df.select(feature_cols).to_numpy()
    y_train_all = train_df.select("label").to_numpy().ravel()
    groups_train_all = train_df.select("group_id").to_numpy().ravel()
    
    X_val_all = val_df.select(feature_cols).to_numpy()
    y_val_all = val_df.select("label").to_numpy().ravel()
    groups_val_all = val_df.select("group_id").to_numpy().ravel()
    
    print(f"   - Kullanılacak güncel özellik sayısı: {len(feature_cols)}")
    
    # 2. GroupKFold Çapraz Doğrulama
    print("\n[2] 5-Fold GroupKFold Çapraz Doğrulama başlatılıyor...")
    gkf = GroupKFold(n_splits=5)
    
    lgb_models = []
    cb_models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_all, y_train_all, groups=groups_train_all)):
        print(f"\n--- FOLD {fold + 1} EĞİTİLİYOR ---")
        X_train, y_train, groups_train = X_train_all[train_idx], y_train_all[train_idx], groups_train_all[train_idx]
        X_val, y_val, groups_val = X_train_all[val_idx], y_train_all[val_idx], groups_train_all[val_idx]
        
        # --- A. LIGHTGBM SINIFLANDIRMA EĞİTİMİ (BINARY) ---
        print("     > LightGBM (CPU - Binary) eğitiliyor...")
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.015,
            'num_leaves': 63,
            'max_depth': 8,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'device': 'cpu',
            'verbose': -1,
            'random_state': 42 + fold,
            'n_jobs': 4
        }
        
        lgb_model = lgb.train(
            lgb_params, train_dataset, num_boost_round=2000,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        lgb_models.append(lgb_model)
        
        # --- B. CATBOOST SIRALAMA EĞİTİMİ (YETIRANK - GPU - Limitlerden Korunmuş) ---
        print("     > CatBoost YetiRank (GPU) eğitiliyor...")
        sort_train_idx = np.argsort(groups_train)
        X_tr_s, y_tr_s, gr_tr_s = X_train[sort_train_idx], y_train[sort_train_idx], groups_train[sort_train_idx]
        
        sort_val_idx = np.argsort(groups_val)
        X_va_s, y_va_s, gr_va_s = X_val[sort_val_idx], y_val[sort_val_idx], groups_val[sort_val_idx]
        
        train_pool = cb.Pool(X_tr_s, label=y_tr_s, group_id=gr_tr_s)
        val_pool = cb.Pool(X_va_s, label=y_va_s, group_id=gr_va_s)
        
        cb_model = cb.CatBoostRanker(
            iterations=2000,
            learning_rate=0.015,
            depth=6,
            loss_function='YetiRank',
            custom_metric=['NDCG', 'MAP'],
            task_type='GPU',            # GPU (CUDA) ARTIK SORUNSUZ ÇALIŞACAK
            random_seed=42 + fold,
            early_stopping_rounds=100,
            verbose=0
        )
        
        try:
            cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            print("       ✔ CatBoost GPU üzerinde başarıyla eğitildi.")
        except Exception as e:
            print(f"       ⚠️ Uyarı: CatBoost GPU hatası verdi ({e}). CPU moduna geçiliyor...")
            cb_model.set_params(task_type='CPU', thread_count=4)
            cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            print("       ✔ CatBoost CPU üzerinde eğitildi.")
            
        cb_models.append(cb_model)
        gc.collect()

    # 3. İzole Doğrulama Kümesi (val_features.csv) Üzerinde Tahmin ve Eşik Optimizasyonu
    print("\n[3] İzole doğrulama kümesi (val_features.csv) tahmin ediliyor...")
    val_preds_lgb = np.zeros(len(val_df))
    val_preds_cb = np.zeros(len(val_df))
    
    for lgb_m, cb_m in zip(lgb_models, cb_models):
        val_preds_lgb += lgb_m.predict(X_val_all) / len(lgb_models)
        val_preds_cb += cb_m.predict(X_val_all) / len(cb_models)
        
    val_preds_lgb_scaled = min_max_scale(val_preds_lgb)
    val_preds_cb_scaled = min_max_scale(val_preds_cb)
    
    # %70 CatBoost + %30 LightGBM ağırlıklı harmanlama
    val_preds_blend = (val_preds_lgb_scaled * 0.3) + (val_preds_cb_scaled * 0.7)
    
    print("[4] Tüm doğrulama kümesi üzerinde En İyi Eşik Değeri (Threshold) aranıyor...")
    best_threshold, best_f1 = find_best_threshold(y_val_all, val_preds_blend)
    val_auc = roc_auc_score(y_val_all, val_preds_blend)
    
    print("\n==================================================")
    print(f"✔ SIZINTISIZ DOĞRULAMA SONUÇLARI (HYBRID ENSEMBLE):")
    print(f"  - En İyi Eşik Değeri (Best Threshold) : {best_threshold:.2f}")
    print(f"  - Sızıntısız Yerel F1 Skoru           : {best_f1:.5f}")
    print(f"  - Sızıntısız Yerel ROC-AUC Skoru      : {val_auc:.5f}")
    print("==================================================\n")
    
    # 4. Test Kümesi Üzerinde Tahmin (Inference - Ensemble)
    print("[5] Test kümesi yükleniyor ve tahminler yapılıyor (Inference)...")
    test_df = pl.read_csv(os.path.join(processed_path, "test_features.csv"))
    test_df = test_df.sort("term_id")
    
    X_test = test_df.select(feature_cols).to_numpy()
    
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_cb = np.zeros(len(test_df))
    
    for lgb_m, cb_m in zip(lgb_models, cb_models):
        test_preds_lgb += lgb_m.predict(X_test) / len(lgb_models)
        test_preds_cb += cb_m.predict(X_test) / len(cb_models)
        
    test_preds_lgb_scaled = min_max_scale(test_preds_lgb)
    test_preds_cb_scaled = min_max_scale(test_preds_cb)
    
    test_preds_prob = (test_preds_lgb_scaled * 0.3) + (test_preds_cb_scaled * 0.7)
    test_preds_binary = (test_preds_prob >= best_threshold).astype(np.int8)
    
    # 5. Submission Hazırlanması
    print("\n[6] Submission (teslimat) dosyası hazırlanıyor...")
    submission = test_df.select("id").with_columns(
        pl.Series("prediction", test_preds_binary)
    )
    
    sub_path = os.path.join(processed_path, "submission.csv")
    submission.write_csv(sub_path)
    print(f"✔ Teslimat dosyası başarıyla diske kaydedildi: {sub_path}")
    print("✔ Sızıntısız Gelişmiş Sıralama Eğitimi ve Test Tahminleri başarıyla tamamlandı!")

if __name__ == "__main__":
    main()