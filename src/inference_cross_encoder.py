import os
import gc
import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Kullanılacak Cihaz: {device.upper()}")

class CrossEncoderInferenceDataset(Dataset):
    """Test/Train çiftlerini BERT için hazırlayan Dataset yapısı"""
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

def predict_and_save(pairs_df, items, terms, model, tokenizer, out_npy_path):
    """Belirli bir çift veri kümesi için BERT Cross-Encoder tahmin olasılıklarını üretip kaydeder."""
    # Metinlerle birleştirme (Join)
    df = pairs_df.join(items, on="item_id", how="left")
    df = df.join(terms, on="term_id", how="left")
    
    df = df.with_columns([
        pl.col("query").fill_null(""),
        pl.col("title").fill_null("")
    ])
    
    queries = df["query"].to_list()
    titles = df["title"].to_list()
    
    del df
    gc.collect()
    
    # DataLoader kurulumu (batch_size=512 ile RTX 4070 için optimize edilmiştir)
    dataset = CrossEncoderInferenceDataset(queries, titles, tokenizer)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    
    probs_list = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Tahmin Ediliyor"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids
                )
            
            logits = outputs.logits
            # Binary sınıflandırmada 1 (Alakalı) sınıfına ait olasılık değerini alıyoruz
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(np.float16)
            probs_list.extend(probs)
            
    # NumPy binary olarak kaydet
    np.save(out_npy_path, np.array(probs_list, dtype=np.float16))
    print(f"    ✔ Tahminler diske başarıyla kaydedildi: {out_npy_path}\n")
    
    del queries, titles, probs_list
    gc.collect()

def main():
    print("=== ADIM 3: SIZINTISIZ BERT CROSS-ENCODER TAHMİNLERİ (INFERENCE) ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    model_path = os.path.join(processed_path, "best_turkish_cross_encoder")
    
    if not os.path.exists(model_path):
        print(f"❌ Hata: Eğitilmiş model bulunamadı: {model_path}")
        return
        
    # Model ve Tokenizer Yükleme
    print("[1] Eğitilmiş Türkçe BERT modeli diske yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = model.to(device)
    model.eval()
    print("    ✔ Model GPU belleğine alındı.")
    
    # Katalog verileri yükleme
    print("\n[2] Katalog verileri yükleniyor...")
    items = pl.read_csv(os.path.join(raw_path, "items.csv")).select(["item_id", "title"])
    terms = pl.read_csv(os.path.join(raw_path, "terms.csv")).select(["term_id", "query"])
    
    # A. Eğitim Kümesi İçin Tahmin Üretme (1.58M satır)
    train_pairs_path = os.path.join(processed_path, "train_pairs_split.csv")
    print(f"\n[3] Sızıntısız Eğitim kümesi işleniyor: {train_pairs_path}")
    train_pairs = pl.read_csv(train_pairs_path)
    train_out_npy = os.path.join(processed_path, "train_bert_cross_sim.npy")
    predict_and_save(train_pairs, items, terms, model, tokenizer, train_out_npy)
    del train_pairs
    gc.collect()
    
    # B. Doğrulama Kümesi İçin Tahmin Üretme (168K satır)
    val_pairs_path = os.path.join(processed_path, "val_pairs_split.csv")
    print(f"[4] Sızıntısız Doğrulama kümesi işleniyor: {val_pairs_path}")
    val_pairs = pl.read_csv(val_pairs_path)
    val_out_npy = os.path.join(processed_path, "val_bert_cross_sim.npy")
    predict_and_save(val_pairs, items, terms, model, tokenizer, val_out_npy)
    del val_pairs
    gc.collect()
    
    # C. Test Kümesi İçin Tahmin Üretme (3.36M satır)
    test_pairs_path = os.path.join(raw_path, "submission_pairs.csv")
    print(f"[5] Test (submission) kümesi işleniyor: {test_pairs_path}")
    test_pairs = pl.read_csv(test_pairs_path)
    test_out_npy = os.path.join(processed_path, "test_bert_cross_sim.npy")
    predict_and_save(test_pairs, items, terms, model, tokenizer, test_out_npy)
    del test_pairs
    gc.collect()
    
    print("=== TÜM TAHMİNLER BAŞARIYLA TAMAMLANDI VE KAYDEDİLDİ ===")

if __name__ == "__main__":
    main()