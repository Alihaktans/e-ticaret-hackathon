import os
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

def find_best_threshold(y_true, y_pred_probs):
    """
    OOF olasılık tahminleri üzerinde tarama yaparak 
    en yüksek F1 skorunu veren en iyi eşik değerini (threshold) bulur.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    
    # 0.01 ile 0.99 arasındaki tüm eşik değerlerini deniyoruz
    for threshold in np.arange(0.01, 1.0, 0.01):
        preds = (y_pred_probs >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
            
    return best_threshold, best_f1

def main():
    print("=== ADIM 6: GELİŞMİŞ MODEL EĞİTİMİ VE F1 SKORU OPTİMİZASYONU ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    processed_path = os.path.join(project_root, "data", "processed")
    
    # 1. Verilerin Yüklenmesi
    print("[1] Eğitim özellikleri yükleniyor...")
    train_df = pl.read_csv(os.path.join(processed_path, "train_features.csv"))
    print(f"✔ {len(train_df):,} eğitim satırı başarıyla yüklendi.")
    
    # Yeni 15 özellikten oluşan gelişmiş listemiz
    feature_cols = [
        'jaccard_sim', 'query_coverage', 'exact_match', 
        'brand_in_query', 'cat_overlap', 'attr_overlap',
        'query_word_len', 'title_word_len', 'len_diff',
        'jaccard_stemmed', 'query_coverage_stemmed',
        'jaccard_3gram', 'query_coverage_3gram',
        'jaccard_4gram', 'query_coverage_4gram'
    ]
    
    X = train_df.select(feature_cols).to_numpy()
    y = train_df.select("label").to_numpy().ravel()
    groups = train_df.select("term_id").to_numpy().ravel()
    
    print(f"   - Kullanılacak gelişmiş özellik sayısı: {len(feature_cols)}")
    
    # 2. GroupKFold Çapraz Doğrulama
    print("\n[2] 5-Fold GroupKFold Çapraz Doğrulama başlatılıyor (term_id bazlı)...")
    gkf = GroupKFold(n_splits=5)
    
    oof_predictions = np.zeros(len(train_df))
    models = []
    feature_importances = np.zeros(len(feature_cols))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- FOLD {fold + 1} EĞİTİLİYOR ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        # Gelişmiş, Kaggle seviyesi hiperparametreler
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.02,       # Daha yavaş ve hassas öğrenme (0.05'ten 0.02'ye düşürüldü)
            'num_leaves': 63,            # Daha karmaşık ilişkileri yakalamak için (31'den 63'e çıkarıldı)
            'max_depth': 8,              # Derinlik artırıldı
            'min_data_in_leaf': 100,     # Aşırı ezberlemeyi (overfitting) engellemek için eklendi
            'feature_fraction': 0.8,     # Özelliklerin %80'ini rastgele seç
            'bagging_fraction': 0.8,     # Satırların %80'ini rastgele seç (genelleme yeteneği için)
            'bagging_freq': 1,
            'verbose': -1,
            'random_state': 42 + fold,   # Çeşitlilik için fold bazlı seed kaydırma
            'n_jobs': -1
        }
        
        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=1500,        # Learning rate düştüğü için ağaç sayısını artırdık
            valid_sets=[train_dataset, val_dataset],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False), # Erken durdurma hassasiyetini artırdık
                lgb.log_evaluation(period=200)
            ]
        )
        
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_preds
        models.append(model)
        
        # Özellik önem derecelerini biriktiriyoruz (Gain tipinde)
        feature_importances += model.feature_importance(importance_type='gain') / 5
        
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_f1_default = f1_score(y_val, (val_preds >= 0.5).astype(int))
        print(f"✔ Fold {fold + 1} | ROC-AUC: {fold_auc:.5f} | Varsayılan F1 (0.50): {fold_f1_default:.5f}")
        
    # 3. Eşik Değeri Optimizasyonu
    print("\n[3] Tüm doğrulama kümesi üzerinde En İyi Eşik Değeri (Threshold) aranıyor...")
    best_threshold, best_f1 = find_best_threshold(y, oof_predictions)
    
    print("\n==================================================")
    print(f"✔ En İyi Eşik Değeri (Best Threshold) : {best_threshold:.2f}")
    print(f"✔ Optimize Edilmiş Yerel F1 Skoru      : {best_f1:.5f}")
    print("==================================================\n")
    
    # Özellik Önem Derecelerini Yazdırma
    print("[Özellik Önem Dereceleri (Gain)]")
    importance_df = pl.DataFrame({
        "Feature": feature_cols,
        "Importance": feature_importances
    }).sort("Importance", descending=True)
    
    for row in importance_df.iter_rows():
        print(f"   - {row[0]:25}: {row[1]:.2f}")
    print()
    
    # 4. Test Tahmini (Inference)
    print("[4] Test kümesi yükleniyor ve tahminler yapılıyor (Inference)...")
    test_df = pl.read_csv(os.path.join(processed_path, "test_features.csv"))
    X_test = test_df.select(feature_cols).to_numpy()
    
    test_preds_prob = np.zeros(len(test_df))
    for model in models:
        test_preds_prob += model.predict(X_test, num_iteration=model.best_iteration) / len(models)
        
    test_preds_binary = (test_preds_prob >= best_threshold).astype(np.int8)
    
    # 5. Submission Hazırlanması
    print("\n[5] Submission (teslimat) dosyası hazırlanıyor...")
    submission = test_df.select("id").with_columns(
        pl.Series("prediction", test_preds_binary)
    )
    
    sub_path = os.path.join(processed_path, "submission.csv")
    submission.write_csv(sub_path)
    print(f"✔ Teslimat dosyası başarıyla diske kaydedildi: {sub_path}")
    print("✔ Model eğitimi ve gelişmiş optimizasyonlar başarıyla tamamlandı!")

if __name__ == "__main__":
    main()