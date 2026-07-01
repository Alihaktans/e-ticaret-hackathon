import os
import gc
import polars as pl
import numpy as np
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

def find_best_threshold(y_true, y_pred_probs):
    """OOF tahminleri üzerinde tarama yaparak en iyi F1 eşik değerini bulur"""
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
    print("=== ADIM 6: HİBRİT GPU-DESTEKLİ TOPLULUK (ENSEMBLE) EĞİTİMİ ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    processed_path = os.path.join(project_root, "data", "processed")
    
    # 1. Verilerin Yüklenmesi
    print("[1] Özellik matrisleri yükleniyor...")
    train_df = pl.read_csv(os.path.join(processed_path, "train_features.csv"))
    print(f"✔ {len(train_df):,} eğitim satırı yüklendi.")
    
    # 21 Özellikten Oluşan Şampiyon Kadro
    feature_cols = [
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed', 'tfidf_sim',
        'color_match', 'material_match',
        'bert_sim_title', 'bert_sim_category', 'bert_sim_attributes', # Çoklu anlamsal benzerlikler
        'jaccard_3gram', 'query_coverage_3gram',
        'jaccard_4gram', 'query_coverage_4gram' # Morfolojik n-gram'lar
    ]
    
    X = train_df.select(feature_cols).to_numpy()
    y = train_df.select("label").to_numpy().ravel()
    groups = train_df.select("term_id").to_numpy().ravel()
    
    print(f"   - Kullanılacak güncel özellik sayısı: {len(feature_cols)}")
    
    # 2. GroupKFold Çapraz Doğrulama
    print("\n[2] 5-Fold GroupKFold Çapraz Doğrulama başlatılıyor...")
    gkf = GroupKFold(n_splits=5)
    
    oof_predictions = np.zeros(len(train_df))
    lgb_models = []
    cb_models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- FOLD {fold + 1} EĞİTİLİYOR ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # --- A. LIGHTGBM EĞİTİMİ (CPU Fallback) ---
        print("     > LightGBM (CPU) eğitiliyor...")
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
            'device': 'cpu', # OpenCL kilitlenmesini engellemek için CPU garantili
            'verbose': -1,
            'random_state': 42 + fold,
            'n_jobs': -1
        }
        
        lgb_model = lgb.train(
            lgb_params, train_dataset, num_boost_round=2000,
            valid_sets=[train_dataset, val_dataset],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
        )
        print("       ✔ LightGBM başarıyla eğitildi.")
            
        lgb_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        lgb_models.append(lgb_model)
        
        # --- B. CATBOOST EĞİTİMİ (GPU) ---
        print("     > CatBoost (GPU) eğitiliyor...")
        cb_model = cb.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.015,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            task_type='GPU',            # GPU (CUDA) AKTİF
            random_seed=42 + fold,
            early_stopping_rounds=100,
            verbose=0
        )
        
        try:
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
            print("       ✔ CatBoost GPU üzerinde eğitildi.")
        except Exception as e:
            print(f"       ⚠️ Uyarı: CatBoost GPU hatası verdi ({e}). CPU moduna geçiliyor...")
            cb_model.set_params(task_type='CPU', thread_count=-1)
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
            print("       ✔ CatBoost CPU üzerinde eğitildi.")
            
        cb_preds = cb_model.predict_proba(X_val)[:, 1]
        cb_models.append(cb_model)
        
        # --- C. HİBRİT TAHMİN BİRLEŞTİRME (LGBM %30 + CatBoost %70) ---
        # Tabular veri setlerinde CatBoost'un kararlılığına %70 ağırlık veriyoruz
        fold_blend_preds = (lgb_preds * 0.3) + (cb_preds * 0.7)
        oof_predictions[val_idx] = fold_blend_preds
        
        fold_auc = roc_auc_score(y_val, fold_blend_preds)
        print(f"       ✔ Fold {fold + 1} Blend ROC-AUC Skoru: {fold_auc:.5f}")
        
        gc.collect()

    # 3. Eşik Değeri Optimizasyonu
    print("\n[3] Tüm doğrulama kümesi üzerinde En İyi Eşik Değeri (Threshold) aranıyor...")
    best_threshold, best_f1 = find_best_threshold(y, oof_predictions)
    
    print("\n==================================================")
    print(f"✔ En İyi Eşik Değeri (Best Threshold) : {best_threshold:.2f}")
    print(f"✔ Optimize Edilmiş Hibrit F1 Skoru     : {best_f1:.5f}")
    print("==================================================\n")
    
    # 4. Test Kümesi Üzerinde Tahmin (Inference - Ensemble)
    print("[4] Test kümesi yükleniyor ve tahminler yapılıyor (Inference)...")
    test_df = pl.read_csv(os.path.join(processed_path, "test_features.csv"))
    X_test = test_df.select(feature_cols).to_numpy()
    
    test_preds_prob = np.zeros(len(test_df))
    for lgb_m, cb_m in zip(lgb_models, cb_models):
        lgb_p = lgb_m.predict(X_test, num_iteration=lgb_m.best_iteration)
        cb_p = cb_m.predict_proba(X_test)[:, 1]
        # Kararlı olması için %70 Catboost, %30 LightGBM birleşimi kullanıyoruz
        test_preds_prob += ((lgb_p * 0.3) + (cb_p * 0.7)) / len(lgb_models)
        
    
    # Modelin kendi bulduğu en iyi eşik değerini (best_threshold = 0.45) kullanıyoruz
    test_preds_binary = (test_preds_prob >= best_threshold).astype(np.int8) 
    
    # 5. Submission Hazırlanması
    print("\n[5] Submission (teslimat) dosyası hazırlanıyor...")
    submission = test_df.select("id").with_columns(
        pl.Series("prediction", test_preds_binary)
    )
    
    sub_path = os.path.join(processed_path, "submission.csv")
    submission.write_csv(sub_path)
    print(f"✔ Teslimat dosyası başarıyla diske kaydedildi: {sub_path}")
    print("✔ Gelişmiş Hibrit Eğitim tamamlandı!")

if __name__ == "__main__":
    main()