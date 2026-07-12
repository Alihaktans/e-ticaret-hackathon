"""
TEKNOFEST TRENDYOL E-TICARET HACKATHONU
Hata Analizi: OOF Tahminlerinden Sistematik Öğrenme
========================================================

train_baseline.py'nin ürettiği oof_predictions.parquet'i items.csv/terms.csv
ile birleştirip yanlış tahminleri (False Positive / False Negative) inceler.
Amaç: "hangi tip hatalar sistematik olarak tekrar ediyor" sorusuna cevap
bulup bir sonraki feature/negatif-örnekleme iyileştirmesine yön vermek.

Kullanım:
    python error_analysis.py --data_dir data/raw --features_dir data/processed2 --out_dir data/processed2
"""

import argparse
import os
import pandas as pd
import numpy as np


def main(data_dir, features_dir, out_dir, top_n=100):
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Veriler yükleniyor...")
    oof = pd.read_parquet(os.path.join(features_dir, "oof_predictions.parquet"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    terms = pd.read_csv(os.path.join(data_dir, "terms.csv"))
    train_feat = pd.read_parquet(os.path.join(features_dir, "train_features.parquet"))

    print("[2/4] Metin bilgileriyle birleştiriliyor...")
    df = oof.merge(terms[["term_id", "query"]], on="term_id", how="left")
    df = df.merge(
        items[["item_id", "title", "category", "brand", "gender", "age_group"]],
        on="item_id", how="left"
    )
    feat_cols = [c for c in train_feat.columns
                 if c not in ["id", "term_id", "item_id", "label"]]
    df = df.merge(train_feat[["term_id", "item_id"] + feat_cols],
                   on=["term_id", "item_id"], how="left")

    df["cat_l1"] = df["category"].fillna("").str.split("/").str[0]
    df["query_word_count"] = df["query"].fillna("").str.split().apply(len)

    # False Positive: gerçekte alakasız (0) ama model alakalı (1) demiş
    # False Negative: gerçekte alakalı (1) ama model alakasız (0) demiş
    fp = df[(df["label"] == 0) & (df["oof_pred_binary"] == 1)].copy()
    fn = df[(df["label"] == 1) & (df["oof_pred_binary"] == 0)].copy()
    tp = df[(df["label"] == 1) & (df["oof_pred_binary"] == 1)]
    tn = df[(df["label"] == 0) & (df["oof_pred_binary"] == 0)]

    print("\n" + "=" * 60)
    print("GENEL HATA DAĞILIMI")
    print("=" * 60)
    print(f"Toplam: {len(df)}  |  TP: {len(tp)}  TN: {len(tn)}  "
          f"FP: {len(fp)}  FN: {len(fn)}")
    print(f"False Positive oranı (tüm negatiflerin içinde): "
          f"{len(fp) / max(len(fp) + len(tn), 1):.4f}")
    print(f"False Negative oranı (tüm pozitiflerin içinde): "
          f"{len(fn) / max(len(fn) + len(tp), 1):.4f}")

    print("\n" + "=" * 60)
    print("HATA ORANI: KATEGORİ (level 1) BAZINDA (en yüksek 15)")
    print("=" * 60)
    cat_stats = df.groupby("cat_l1").apply(
        lambda g: pd.Series({
            "n": len(g),
            "error_rate": ((g["label"] != g["oof_pred_binary"]).sum()) / len(g),
        })
    ).sort_values("error_rate", ascending=False)
    print(cat_stats[cat_stats["n"] >= 50].head(15).to_string())

    print("\n" + "=" * 60)
    print("HATA ORANI: QUERY UZUNLUĞU BAZINDA")
    print("=" * 60)
    df["qlen_bucket"] = pd.cut(df["query_word_count"], bins=[0, 1, 2, 3, 4, 100],
                                 labels=["1", "2", "3", "4", "5+"])
    qlen_stats = df.groupby("qlen_bucket", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "error_rate": ((g["label"] != g["oof_pred_binary"]).sum()) / len(g),
        })
    )
    print(qlen_stats.to_string())

    print("\n" + "=" * 60)
    print("FALSE POSITIVE ÖRNEKLERİ (model 'alakalı' dedi ama değildi) -- "
          f"en yüksek güvenle yanılanlar (top {min(15, len(fp))})")
    print("=" * 60)
    fp_sorted = fp.sort_values("oof_pred", ascending=False)
    for _, r in fp_sorted.head(15).iterrows():
        print(f"  query='{r['query']}' | title='{r['title']}' | "
              f"cat={r['category']} | oof_pred={r['oof_pred']:.3f} | "
              f"tfidf_cos={r.get('tfidf_cosine', float('nan')):.3f}")

    print("\n" + "=" * 60)
    print("FALSE NEGATIVE ÖRNEKLERİ (model 'alakasız' dedi ama alakalıydı) -- "
          f"en düşük güvenle yanılanlar (top {min(15, len(fn))})")
    print("=" * 60)
    fn_sorted = fn.sort_values("oof_pred", ascending=True)
    for _, r in fn_sorted.head(15).iterrows():
        print(f"  query='{r['query']}' | title='{r['title']}' | "
              f"cat={r['category']} | oof_pred={r['oof_pred']:.3f} | "
              f"tfidf_cos={r.get('tfidf_cosine', float('nan')):.3f}")

    # Kaydet: sonraki manuel inceleme için CSV çıktıları
    fp_sorted.head(top_n).to_csv(os.path.join(out_dir, "false_positives_top.csv"), index=False)
    fn_sorted.head(top_n).to_csv(os.path.join(out_dir, "false_negatives_top.csv"), index=False)
    cat_stats.to_csv(os.path.join(out_dir, "error_rate_by_category.csv"))

    print(f"\n[3/4] false_positives_top.csv, false_negatives_top.csv, "
          f"error_rate_by_category.csv kaydedildi -> {out_dir}")

    print("\n[4/4] TAMAMLANDI. Şimdi false_positives_top.csv ve "
          "false_negatives_top.csv dosyalarını manuel gözden geçirip "
          "tekrar eden hata kalıplarını (eş anlamlılık, kategori "
          "karışıklığı, marka/attribute yanlış eşleşme vb.) not alın.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--features_dir", default="data/processed2")
    parser.add_argument("--out_dir", default="data/processed2")
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()
    main(args.data_dir, args.features_dir, args.out_dir, args.top_n)