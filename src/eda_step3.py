import os
import pandas as pd
import re

def clean_text(text):
    """Basit bir temizleme fonksiyonu: Küçük harfe çevirme ve noktalama temizliği"""
    if pd.isna(text):
        return ""
    # Türkçe küçük harf dönüşümü için özel kurallar
    text = str(text).replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    # Noktalama işaretlerini boşlukla değiştirme
    text = re.sub(r'[^\w\s]', ' ', text)
    # Çift boşlukları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_step3_text_analysis():
    print("=== ADIM 3: METİNSEL ANALİZ VE SÖZCÜK ÖRTÜŞME ORANLARI ===\n")
    
    # 1. Dosya Yollarını Otomatik Çözümleme
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "raw")
    
    items_path = os.path.join(data_path, "items.csv")
    terms_path = os.path.join(data_path, "terms.csv")
    train_path = os.path.join(data_path, "training_pairs.csv")
    
    # Dosya kontrolleri
    for p in [items_path, terms_path, train_path]:
        if not os.path.exists(p):
            print(f"❌ Hata: Dosya bulunamadı: {p}")
            return

    print("[1] Dosyalar yükleniyor...")
    items = pd.read_csv(items_path, usecols=["item_id", "title"])
    terms = pd.read_csv(terms_path)
    train = pd.read_csv(train_path)
    print("✔ Dosyalar yüklendi.\n")

    # 2. Karakter ve Kelime Sayısı Dağılımı
    print("[2] Metin Uzunluğu İstatistikleri:")
    
    # Sorgular (Queries)
    terms['word_count'] = terms['query'].fillna("").apply(lambda x: len(str(x).split()))
    terms['char_count'] = terms['query'].fillna("").apply(len)
    
    # Ürün Başlıkları (Titles)
    items['word_count'] = items['title'].fillna("").apply(lambda x: len(str(x).split()))
    items['char_count'] = items['title'].fillna("").apply(len)
    
    print("   - Arama Sorguları (terms.csv):")
    print(f"     * Ortalama Kelime Sayısı   : {terms['word_count'].mean():.2f} (En çok: {terms['word_count'].max()})")
    print(f"     * Ortalama Karakter Sayısı : {terms['char_count'].mean():.2f} (En çok: {terms['char_count'].max()})")
    
    print("   - Ürün Başlıkları (items.csv):")
    print(f"     * Ortalama Kelime Sayısı   : {items['word_count'].mean():.2f} (En çok: {items['word_count'].max()})")
    print(f"     * Ortalama Karakter Sayısı : {items['char_count'].mean():.2f} (En çok: {items['char_count'].max()})")
    print()

    # 3. Gerçek Eşleşmelerdeki Sözcük Örtüşme (Token Overlap) Analizi
    print("[3] Pozitif Eğitim Çiftlerindeki Sözcük Örtüşme Analizi:")
    print("   - Eşleşmeler birleştiriliyor (Merge)...")
    
    # train veri kümesini items ve terms ile birleştiriyoruz
    merged = train.merge(items, on="item_id", how="left")
    merged = merged.merge(terms, on="term_id", how="left")
    
    print("   - Metinler ön işlemden geçiriliyor...")
    merged['clean_query'] = merged['query'].apply(clean_text)
    merged['clean_title'] = merged['title'].apply(clean_text)
    
    # Örtüşme metriklerini hesaplama fonksiyonları
    def calculate_overlap_metrics(row):
        q_words = set(row['clean_query'].split())
        t_words = set(row['clean_title'].split())
        
        if not q_words or not t_words:
            return 0.0, 0.0, 0.0
            
        intersection = q_words.intersection(t_words)
        
        # 1. Jaccard Benzerliği (Ortak Kelime / Toplam Farklı Kelime)
        jaccard = len(intersection) / len(q_words.union(t_words))
        
        # 2. Sorgudaki kelimelerin yüzde kaçı başlıkta geçiyor? (Kapsama Oranı)
        query_coverage = len(intersection) / len(q_words)
        
        # 3. Tam eşleşme (Sorgunun tamamı başlıkta bir bütün olarak geçiyor mu?)
        exact_match = 1.0 if row['clean_query'] in row['clean_title'] else 0.0
        
        return jaccard, query_coverage, exact_match

    print("   - Metrikler hesaplanıyor (Bu işlem veri boyutundan dolayı yarım dakika kadar sürebilir)...")
    metrics = merged.apply(calculate_overlap_metrics, axis=1)
    
    merged['jaccard'] = [m[0] for m in metrics]
    merged['query_coverage'] = [m[1] for m in metrics]
    merged['exact_match'] = [m[2] for m in metrics]
    
    print(f"     * Ortalama Jaccard Benzerliği                       : {merged['jaccard'].mean():.4f}")
    print(f"     * Sorgu Kelimelerinin Başlıkta Geçme Oranı (Kapsama): %{merged['query_coverage'].mean()*100:.2f}")
    print(f"     * Sorgunun Başlıkta Birebir Alt Metin Olarak Geçme Oranı: %{merged['exact_match'].mean()*100:.2f}")
    print()

    # 4. En Düşük Örtüşmeye Sahip İlginç Pozitif Eşleşme Örnekleri
    # Örtüşme oranı çok düşük olmasına rağmen "Alakalı (1)" etiketlenmiş örnekler anlamsal (semantic) eşleşmeleri gösterir.
    print("[4] Düşük Sözcük Örtüşmesine Sahip İlginç Pozitif Örnekler:")
    semantic_samples = merged[merged['query_coverage'] == 0.0].head(5)
    
    if len(semantic_samples) > 0:
        for idx, row in semantic_samples.iterrows():
            print(f"     - Sorgu: '{row['query']}'  ==>  Ürün Başlığı: '{row['title']}'")
    else:
        print("     - Tüm pozitif örneklerde en az 1 kelime ortak.")
        
    print("\n=== ANALİZ TAMAMLANDI ===")

if __name__ == "__main__":
    run_step3_text_analysis()