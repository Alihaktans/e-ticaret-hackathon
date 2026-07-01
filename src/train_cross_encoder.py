import os
import gc
import torch
import numpy as np
import polars as pl
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    """F1 skorunu hesaplayan metrik fonksiyonu"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds := predictions)
    return {"f1": f1}

def main():
    print("=== COSNUP: MASTER-LEVEL TÜRKÇE BERT FINE-TUNING PIPELINE ===\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_path = os.path.join(project_root, "data", "raw")
    processed_path = os.path.join(project_root, "data", "processed")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Kullanılacak Cihaz: {device.upper()}")
    if device == "cuda":
        print(f"    - Ekran Kartı: {torch.cuda.get_device_name(0)}")
        
    # 1. Model ve Tokenizer Yükleme
    model_name = "dbmdz/bert-base-turkish-cased"
    print(f"\n[2] Türkçe BERT tokenizer yükleniyor: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Verilerin Yüklenmesi ve Birleştirilmesi
    print("\n[3] Eğitim verileri Polars ile yükleniyor... (Join)")
    train_pairs = pl.read_csv(os.path.join(processed_path, "train_with_negatives.csv"))
    items = pl.read_csv(os.path.join(raw_path, "items.csv")).select(["item_id", "title"])
    terms = pl.read_csv(os.path.join(raw_path, "terms.csv")).select(["term_id", "query"])
    
    train_df = train_pairs.join(items, on="item_id", how="left")
    train_df = train_df.join(terms, on="term_id", how="left")
    
    train_df = train_df.with_columns([
        pl.col("query").fill_null(""),
        pl.col("title").fill_null("")
    ])
    
    # Karıştırma ve Holdout Ayrımı
    train_df = train_df.sample(fraction=1.0, seed=42)
    split_idx = int(len(train_df) * 0.9)
    
    print("   - Veriler yüksek performanslı PyArrow tablolarına dönüştürülüyor...")
    train_pd = train_df[0:split_idx].select(["query", "title", "label"]).to_pandas()
    val_pd = train_df[split_idx:].select(["query", "title", "label"]).to_pandas()
    
    train_dataset = HFDataset.from_pandas(train_pd)
    val_dataset = HFDataset.from_pandas(val_pd)
    
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    
    # Bellek temizliği
    del train_df, train_pd, val_pd, train_pairs, items, terms
    gc.collect()
    
    # 3. Çoklu İşlemci Çekirdekleriyle Paralel Tokenizasyon
    print("\n[4] Tüm çekirdekler kullanılarak paralel ön-tokenizasyon yapılıyor...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["query"],
            examples["title"],
            truncation=True,
            max_length=128
        )
    
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=os.cpu_count(), 
        remove_columns=["query", "title"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=os.cpu_count(), 
        remove_columns=["query", "title"]
    )
    
    # 4. Modelin Yüklenmesi
    print(f"\n[5] Sınıflandırma kafası eklenerek model GPU'ya yükleniyor...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 5. Süper-Optimize Eğitim Parametreleri (Gereksiz group_by_length parametresi kaldırıldı)
    print("\n[6] Eğitim parametreleri ayarlanıyor...")
    training_args = TrainingArguments(
        output_dir=os.path.join(processed_path, "bert_cross_encoder"),
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2,
        logging_steps=500,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_num_workers=4,
        report_to="none"
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 6. Eğitim Başlangıcı
    print("\n[7] Türkçe BERT Fine-Tuning Başlatılıyor...")
    print("    - Bu işlem ekran kartınızda yaklaşık 1.5 - 2.5 saat sürecektir.")
    print("    - Eğitim boyunca ekran kartınız %100 yük altında çalışacaktır.\n")
    
    trainer.train()
    
    # En iyi modeli kaydetme
    model_save_path = os.path.join(processed_path, "best_turkish_cross_encoder")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\n✔ En iyi model başarıyla diske kaydedildi: {model_save_path}")
    print("=== FINE-TUNING TAMAMLANDI ===")

if __name__ == "__main__":
    main()