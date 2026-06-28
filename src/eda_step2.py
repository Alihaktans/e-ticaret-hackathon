import os
import pandas as pd
import json

def run_step2_analysis():
    print("=== ADIM 2: EKSİK, BİLİNMEYEN DEĞERLER VE KATALOG DETAY ANALİZİ ===\n")
    
    # 1. Dosya Yollarını Otomatik Çözümleme
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "raw")
    items_path = os.path.join(data_path, "items.csv")
    
    if not os.path.exists(items_path):
        print(f"❌ Hata: items.csv dosyası bulunamadı: {items_path}")
        return

    print("[1] items.csv yükleniyor...")
    items = pd.read_csv(items_path)
    total_rows = len(items)
    print(f"✔ {total_rows} ürün başarıyla yüklendi.\n")

    # 2. Genel Eksik (NaN) Değer Analizi
    print("[2] Genel Eksik (NaN) Değer Dağılımı:")
    nan_counts = items.isna().sum()
    for col in items.columns:
        nan_cnt = nan_counts[col]
        nan_pct = (nan_cnt / total_rows) * 100
        print(f"   - {col:12}: {nan_cnt:<7} adet NaN (%{nan_pct:.2f})")
    print()

    # 3. Gizli Eksik Değer Analizi (unknown vb. durumlar)
    print("[3] Gizli Bilinmeyen (unknown, boş string vb.) Değer Analizi:")
    # İncelemek istediğimiz kategorik kolonlar
    target_cols = ['brand', 'gender', 'age_group', 'category']
    
    for col in target_cols:
        if col in items.columns:
            # Sütundaki değerleri küçük harfe çevirip analiz ediyoruz
            col_cleaned = items[col].fillna("MISSING_VALUE").astype(str).str.lower().str.strip()
            
            # Bilinmeyen kabul edilebilecek kelimeleri arıyoruz
            unknown_keywords = ['unknown', 'bilinmiyor', 'belirtilmemiş', 'nan', '', 'missing_value']
            unknown_mask = col_cleaned.isin(unknown_keywords)
            unknown_cnt = unknown_mask.sum()
            unknown_pct = (unknown_cnt / total_rows) * 100
            
            print(f"   - {col:12}: {unknown_cnt:<7} adet bilinmeyen/boş değer (%{unknown_pct:.2f})")
            
            # Boş olmayan en popüler ilk 3 değeri yazdıralım
            top_values = items[~items[col].isna() & ~items[col].astype(str).str.lower().str.strip().isin(unknown_keywords)][col].value_counts().head(3)
            print(f"     * En popüler değerler:")
            for val, count in top_values.items():
                print(f"       > {val}: {count} adet (%{count/total_rows*100:.1f})")
    print()

    # 4. Kategori Hiyerarşisi Derinlik Analizi
    print("[4] Kategori Ağacı Derinlik Analizi:")
    # Kategori zincirindeki '/' karakterlerini sayarak derinlik buluyoruz
    # Örn: 'ayakkabı/spor/sneaker' -> 2 adet '/' içerir, yani 3 seviyelidir.
    valid_categories = items['category'].dropna()
    depths = valid_categories.apply(lambda x: len(str(x).split('/')))
    
    print(f"   - En sığ kategori seviyesi : {depths.min()}")
    print(f"   - En derin kategori seviyesi: {depths.max()}")
    print(f"   - Ortalama kategori derinliği: {depths.mean():.2f}")
    print("   - Seviye dağılımları:")
    for level, count in depths.value_counts().sort_index().items():
        print(f"     * {level} seviyeli kategoriler: {count:<6} adet (%{count/len(valid_categories)*100:.1f})")
    print()

    # 5. Attributes (Özellikler) Alanı Analizi
    print("[5] Attributes Özellik Yapısı Analizi:")
    valid_attributes = items['attributes'].dropna()
    print(f"   - Dolu attribute içeren ürün oranı: %{len(valid_attributes)/total_rows*100:.1f}")
    
    # İlk 3 dolu attribute satırını örnek olarak gösterelim
    if len(valid_attributes) > 0:
        print("   - Örnek attribute formatları:")
        for i, sample in enumerate(valid_attributes.head(3)):
            print(f"     {i+1}. Örnek: {sample[:120]}...")
            
    print("\n=== ANALİZ TAMAMLANDI ===")

if __name__ == "__main__":
    run_step2_analysis()