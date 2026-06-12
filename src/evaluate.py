# -*- coding: utf-8 -*-
"""
Eğitilmiş CRF modellerini test kümesinde değerlendirir.

Hesaplanan metrikler (proje isterleri madde 4):
  - Öbek (entity) düzeyinde precision / recall / f1-measure  (seqeval)
  - Kelime (token) düzeyinde accuracy
  - Her sınıf için precision/recall/f1 (sklearn classification_report)
  - Her sınıf için karışıklık matrisi (confusion matrix)  -> grafik (PNG)
  - Sınıf bazında F1 çubuk grafiği                        -> grafik (PNG)

Çıktılar results/ klasörüne yazılır.

Kullanım:
  python src/evaluate.py                 # chunk + clause + pos
  python src/evaluate.py --task chunk
  python src/evaluate.py --task chunk --end2end   # POS'u da model tahmin etsin
"""
import os
import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score)
from seqeval.metrics import (precision_score, recall_score, f1_score,
                             classification_report as seq_report)

from features import read_conll, sent2features, sent2labels

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "chunks")
MODELS = os.path.join(HERE, "..", "models")
RESULTS = os.path.join(HERE, "..", "results")

TASKS = {
    "pos":    (1, False, "crf_pos.pkl", "POS Etiketleme"),
    "chunk":  (2, True,  "crf_chunk.pkl", "Chunking (Öbekleme)"),
    "clause": (3, True,  "crf_clause.pkl", "Clause (Yan Cümlecik)"),
}


def load(fname):
    with open(os.path.join(MODELS, fname), "rb") as f:
        return pickle.load(f)


def plot_confusion(y_true_flat, y_pred_flat, labels, title, out_png):
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.7),
                                    max(5, len(labels) * 0.6)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Tahmin edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title(title + " - Karışıklık Matrisi (normalize)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("    grafik kaydedildi:", os.path.relpath(out_png))


def plot_f1_bars(report_dict, title, out_png):
    classes = [k for k in report_dict
               if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")]
    f1s = [report_dict[c]["f1-score"] for c in classes]
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.6), 4))
    bars = ax.bar(range(len(classes)), f1s, color="#4C72B0")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(title + " - Sınıf Bazında F1")
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.2f" % v,
                ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print("    grafik kaydedildi:", os.path.relpath(out_png))


def evaluate_task(task, test_sents, end2end=False, pos_model=None):
    col, use_upos, fname, title = TASKS[task]
    crf = load(fname)

    # özellikler: end2end ise POS'u pos_model tahmin eder
    X, y_true = [], []
    for s in test_sents:
        pred_pos = None
        if use_upos and end2end and pos_model is not None:
            xf = sent2features(s, use_upos=False)
            pred_pos = pos_model.predict_single(xf)
        X.append(sent2features(s, use_upos=use_upos, use_pred_pos=pred_pos))
        y_true.append(sent2labels(s, col))

    y_pred = crf.predict(X)

    # düz (flat) listeler
    yt = [l for seq in y_true for l in seq]
    yp = [l for seq in y_pred for l in seq]

    os.makedirs(RESULTS, exist_ok=True)
    lines = []
    lines.append("=" * 60)
    lines.append("GÖREV: %s%s" % (title, "  [uçtan uca / tahmini POS]" if end2end else ""))
    lines.append("=" * 60)
    lines.append("Test cümlesi : %d" % len(test_sents))
    lines.append("Test kelimesi: %d" % len(yt))
    lines.append("Token doğruluğu: %.4f" % accuracy_score(yt, yp))

    if task in ("chunk", "clause"):
        # entity (öbek) düzeyi - seqeval BIO bekler
        lines.append("")
        lines.append("--- ÖBEK (ENTITY) DÜZEYİ [seqeval] ---")
        lines.append("Precision: %.4f" % precision_score(y_true, y_pred))
        lines.append("Recall   : %.4f" % recall_score(y_true, y_pred))
        lines.append("F1-measure: %.4f" % f1_score(y_true, y_pred))
        lines.append("")
        lines.append(seq_report(y_true, y_pred, digits=4))

    # token düzeyi sınıf bazında rapor
    labels = sorted(set(yt) | set(yp))
    rep_txt = classification_report(yt, yp, labels=labels, digits=4, zero_division=0)
    rep_dict = classification_report(yt, yp, labels=labels, digits=4,
                                     zero_division=0, output_dict=True)
    lines.append("")
    lines.append("--- KELİME (TOKEN) DÜZEYİ sınıf bazında ---")
    lines.append(rep_txt)

    report_text = "\n".join(lines)
    print(report_text)
    suffix = "_end2end" if end2end else ""
    with open(os.path.join(RESULTS, "metrics_%s%s.txt" % (task, suffix)),
              "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    # grafikler
    plot_confusion(yt, yp, labels, title,
                   os.path.join(RESULTS, "confusion_%s%s.png" % (task, suffix)))
    plot_f1_bars(rep_dict, title,
                 os.path.join(RESULTS, "f1_%s%s.png" % (task, suffix)))
    return rep_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    ap.add_argument("--end2end", action="store_true",
                    help="UPOS'u gold yerine POS modeliyle tahmin et (chunk/clause)")
    args = ap.parse_args()

    test_sents = read_conll(os.path.join(DATA, "test.conll"))
    pos_model = load("crf_pos.pkl") if args.end2end else None

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        e2e = args.end2end and t in ("chunk", "clause")
        evaluate_task(t, test_sents, end2end=e2e, pos_model=pos_model)
        print()


if __name__ == "__main__":
    main()

