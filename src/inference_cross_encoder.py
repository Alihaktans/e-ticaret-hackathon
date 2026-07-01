import os
import gc
import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# CUDA (GPU) Kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılacak Cihaz: {device.upper()}")

class CrossEncoderInferenceDataset(Dataset):
    """Test verilerini BERT'e beslemek üzere hazırlayan PyTorch Dataset yapısı"""
    def __init__(self, queries, titles, tokenizer, max_len=96):
        self.queries = queries
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.queries[idx],
            self.titles[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

def main():
    print("=== ADIM 8: TÜRKÇE BERT CROSS-ENCODER TEST TAHMİNİ (INFERENCE) ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    model_path = os.path.join(processed_path, "best_turkish_cross_encoder")
    
    if not os.path.exists(model_path):
        print(f"❌ Hata: Eğitilmiş model bulunamadı: {model_path}")
        return
        
    # 1. Model ve Tokenizer Yükleme
    print("[1] Eğitilmiş Türkçe BERT modeli diske yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = model.to(device)
    model.eval() # Modeli değerlendirme moduna alıyoruz (dropout vb. kapanır)
    print("    ✔ Model GPU belleğine alındı.")
    
    # 2. Test Verilerinin Yüklenmesi ve Hazırlanması
    print("\n[2] Test verileri Polars ile yükleniyor...")
    test_pairs = pl.read_csv(os.path.join(raw_path, "submission_pairs.csv"))
    items = pl.read_csv(os.path.join(raw_path, "items.csv")).select(["item_id", "title"])
    terms = pl.read_csv(os.path.join(raw_path, "terms.csv")).select(["term_id", "query"])
    
    # Test çiftlerini metinlerle birleştiriyoruz
    test_df = test_pairs.join(items, on="item_id", how="left")
    test_df = test_df.join(terms, on="term_id", how="left")
    
    test_df = test_df.with_columns([
        pl.col("query").fill_null(""),
        pl.col("title").fill_null("")
    ])
    
    test_ids = test_df["id"].to_list()
    queries = test_df["query"].to_list()
    titles = test_df["title"].to_list()
    
    # Belleği rahatlatmak için dataframe'leri siliyoruz
    del test_pairs, items, terms, test_df
    gc.collect()
    
    # 3. Dataset ve DataLoader Tanımlaması
    print("\n[3] PyTorch DataLoader hazırlanıyor (RTX 4070 için optimize edildi)...")
    inference_dataset = CrossEncoderInferenceDataset(queries, titles, tokenizer)
    
    # batch_size=512 ve num_workers=4 ile veri akışını son derece hızlandırıyoruz
    inference_loader = DataLoader(
        inference_dataset, 
        batch_size=512, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 4. GPU-Hızlandırmalı Tahmin Döngüsü
    print(f"\n[4] 3,359,679 satır için tahminler üretiliyor...")
    print("    - FP16 (Yarı hassasiyet) aktif edilerek işlem hızı maksimuma çıkarıldı.")
    print("    - Tahmin süresi: ~25 - 35 dakika.\n")
    
    predictions = []
    
    # Gradyan hesaplamalarını kapatarak bellekten ve işlemden devasa tasarruf sağlıyoruz
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Tahmin Ediliyor"):
            # Verileri GPU'ya taşıyoruz
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            
            # Autocast ile FP16 yarı hassasiyette hızlı forward pass yapıyoruz
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids
                )
            
            # Logit değerlerini olasılığa dönüştürme ve argmax ile doğrudan 0 veya 1 etiketini alma
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int8)
            predictions.extend(batch_predictions)
            
    # 5. Teslimat (Submission) Dosyasının Kaydedilmesi
    print("\n[5] Submission (teslimat) dosyası hazırlanıyor...")
    submission = pl.DataFrame({
        "id": test_ids,
        "prediction": predictions
    })
    
    sub_path = os.path.join(processed_path, "submission.csv")
    submission.write_csv(sub_path)
    print(f"✔ Teslimat dosyası başarıyla kaydedildi: {sub_path}")
    print("=== TAHMİN SÜRECİ TAMAMLANDI ===")

if __name__ == "__main__":
    main()