# -*- coding: utf-8 -*-
"""
CRF modellerini eğitir ve models/ klasörüne kaydeder.

Üç model eğitilir:
  1) POS  modeli (col=upos)   : yalnızca yüzeysel özellikler -> ham metni POS'lar
  2) CHUNK modeli (col=chunk) : yüzeysel + UPOS özellikleri (ana görev)
  3) CLAUSE modeli (col=clause): yüzeysel + UPOS özellikleri (yan cümlecik katmanı)

Kullanım:
  python src/train.py            # her üç modeli de eğitir
  python src/train.py --task chunk
"""
import os
import argparse
import pickle

import sklearn_crfsuite

from features import read_conll, sent2features, sent2labels

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "chunks")
MODELS = os.path.join(HERE, "..", "models")

# task -> (kolon indeksi, UPOS özelliği kullanılsın mı, model dosyası)
TASKS = {
    "pos":    (1, False, "crf_pos.pkl"),
    "chunk":  (2, True,  "crf_chunk.pkl"),
    "clause": (3, True,  "crf_clause.pkl"),
}


def train_one(task, train_sents):
    col, use_upos, fname = TASKS[task]
    X = [sent2features(s, use_upos=use_upos) for s in train_sents]
    y = [sent2labels(s, col) for s in train_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=120,
        all_possible_transitions=True,
    )
    crf.fit(X, y)

    os.makedirs(MODELS, exist_ok=True)
    out = os.path.join(MODELS, fname)
    with open(out, "wb") as f:
        pickle.dump(crf, f)
    print("  [%s] egitildi  ->  %s  (%d ozellik, %d etiket)" % (
        task, fname, crf.num_attributes_ if hasattr(crf, "num_attributes_") else -1,
        len(crf.classes_)))
    return crf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    args = ap.parse_args()

    train_sents = read_conll(os.path.join(DATA, "train.conll"))
    print("Egitim verisi: %d cumle" % len(train_sents))

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        train_one(t, train_sents)
    print("Egitim tamam ->", os.path.abspath(MODELS))


if __name__ == "__main__":
    main()

