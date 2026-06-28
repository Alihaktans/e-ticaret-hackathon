import os
import pandas as pd

def run_step1_profiling():
    print("=== ADIM 1: GENEL PROFİL VE BENZERSİZ DEĞER ANALİZİ ===\n")
    
    # 1. Dosya Yollarını Otomatik Çözümleme
    # Bu dosyanın (eda_step1.py) bulunduğu dizin: src/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Bir üst dizine çıkıp (proje kök dizini) data/raw klasörüne gidiyoruz
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "raw")
    
    print(f"[1] Veri klasörü aranıyor: {data_path}")
    
    # Gerekli dosyaların yolları
    files = {
        "items": os.path.join(data_path, "items.csv"),
        "terms": os.path.join(data_path, "terms.csv"),
        "train": os.path.join(data_path, "training_pairs.csv"),
        "test": os.path.join(data_path, "submission_pairs.csv")
    }
    
    # Dosya varlık kontrolü
    missing_files = []
    for name, path in files.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}.csv ({path})")
            
    if missing_files:
        print("\n❌ Hata: Aşağıdaki dosyalar belirtilen konumda bulunamadı:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nLütfen dosyaların 'data/raw/' klasöründe olduğundan emin olun.")
        return

    # Verileri Yükleme
    print("✔ Gerekli tüm dosyalar bulundu. Yükleniyor...")
    items = pd.read_csv(files["items"])
    terms = pd.read_csv(files["terms"])
    train = pd.read_csv(files["train"])
    test = pd.read_csv(files["test"])
    print("✔ Tüm dosyalar başarıyla belleğe yüklendi.\n")

    # 2. Boyut ve Bellek Analizi
    print("[2] Boyut ve Bellek Analizi:")
    for name, df in zip(["items", "terms", "train_pairs", "test_pairs"], [items, terms, train, test]):
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"   - {name:12}: Satır Sayısı: {df.shape[0]:<8} | Sütun Sayısı: {df.shape[1]:<2} | Bellek: {memory_usage_mb:.2f} MB")
    print()

    # 3. Benzersiz (Unique) Değer Sayıları
    print("[3] Benzersiz Kimlik (Unique ID) Sayıları:")
    unique_items_catalog = items["item_id"].nunique()
    unique_terms_catalog = terms["term_id"].nunique()
    print(f"   - Katalogdaki toplam benzersiz ürün sayısı (items): {unique_items_catalog}")
    print(f"   - Katalogdaki toplam benzersiz terim sayısı (terms): {unique_terms_catalog}")
    
    train_unique_items = train["item_id"].nunique()
    train_unique_terms = train["term_id"].nunique()
    test_unique_items = test["item_id"].nunique()
    test_unique_terms = test["term_id"].nunique()
    
    print(f"   - Eğitim kümesindeki (train) benzersiz ürün sayısı: {train_unique_items} (%{train_unique_items/unique_items_catalog*100:.1f})")
    print(f"   - Eğitim kümesindeki (train) benzersiz terim sayısı: {train_unique_terms} (%{train_unique_terms/unique_terms_catalog*100:.1f})")
    print(f"   - Test kümesindeki (test) benzersiz ürün sayısı    : {test_unique_items} (%{test_unique_items/unique_items_catalog*100:.1f})")
    print(f"   - Test kümesindeki (test) benzersiz terim sayısı    : {test_unique_terms} (%{test_unique_terms/unique_terms_catalog*100:.1f})")
    print()

    # 4. Overlap (Görülmemiş Değer/Sızma) Analizi
    print("[4] Overlap ve Sızma (Leakage) Analizi:")
    
    train_terms_set = set(train["term_id"])
    test_terms_set = set(test["term_id"])
    seen_terms_in_test = test_terms_set.intersection(train_terms_set)
    unseen_terms_in_test = test_terms_set - train_terms_set
    
    train_items_set = set(train["item_id"])
    test_items_set = set(test["item_id"])
    seen_items_in_test = test_items_set.intersection(train_items_set)
    unseen_items_in_test = test_items_set - train_items_set
    
    print(f"   - Test kümesindeki terimlerin ne kadarı eğitimde var (Seen)?   : {len(seen_terms_in_test)} (%{len(seen_terms_in_test)/len(test_terms_set)*100:.1f})")
    print(f"   - Test kümesindeki terimlerin ne kadarı eğitimde YOK (Unseen)?  : {len(unseen_terms_in_test)} (%{len(unseen_terms_in_test)/len(test_terms_set)*100:.1f})")
    print(f"   - Test kümesindeki ürünlerin ne kadarı eğitimde var (Seen)?      : {len(seen_items_in_test)} (%{len(seen_items_in_test)/len(test_items_set)*100:.1f})")
    print(f"   - Test kümesindeki ürünlerin ne kadarı eğitimde YOK (Unseen)?    : {len(unseen_items_in_test)} (%{len(unseen_items_in_test)/len(test_items_set)*100:.1f})")
    print()

    # 5. Bütünlük (Integrity) Kontrolü
    print("[5] Veri Bütünlüğü Kontrolü:")
    catalog_items_set = set(items["item_id"])
    catalog_terms_set = set(terms["term_id"])
    
    missing_items_in_train = train_items_set - catalog_items_set
    missing_terms_in_train = train_terms_set - catalog_terms_set
    missing_items_in_test = test_items_set - catalog_items_set
    missing_terms_in_test = test_terms_set - catalog_terms_set
    
    print(f"   - Katalogda bulunmayan ama train setinde geçen ürün sayısı: {len(missing_items_in_train)}")
    print(f"   - Katalogda bulunmayan ama train setinde geçen terim sayısı: {len(missing_terms_in_train)}")
    print(f"   - Katalogda bulunmayan ama test setinde geçen ürün sayısı : {len(missing_items_in_test)}")
    print(f"   - Katalogda bulunmayan ama test setinde geçen terim sayısı : {len(missing_terms_in_test)}")
    print("\n=== ANALİZ TAMAMLANDI ===")

if __name__ == "__main__":
    run_step1_profiling()