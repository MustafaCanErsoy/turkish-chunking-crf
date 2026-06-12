# -*- coding: utf-8 -*-
"""
Uçtan uca demo: Ham bir Türkçe cümle -> POS -> CHUNK -> CLAUSE (CoNLL çıktısı).

Kullanım:
  python src/predict.py "Dün akşam toplantıdan erken çıkan öğrenci eve gitti."
  python src/predict.py            # dosyadaki örnek cümleyi çalıştırır
"""
import os
import sys
import re
import pickle

from features import sent2features

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models")

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def load(fname):
    with open(os.path.join(MODELS, fname), "rb") as f:
        return pickle.load(f)


def tokenize(text):
    return TOKEN_RE.findall(text)


def predict(text, pos_crf, chunk_crf, clause_crf):
    tokens = tokenize(text)
    # geçici cümle yapısı: (form, upos_placeholder, _, _)
    sent = [(t, "_", "_", "_") for t in tokens]

    # 1) POS (yüzeysel özellikler)
    pos = pos_crf.predict_single(sent2features(sent, use_upos=False))
    # 2) CHUNK (tahmini POS ile)
    feats = sent2features(sent, use_upos=True, use_pred_pos=pos)
    chunk = chunk_crf.predict_single(feats)
    # 3) CLAUSE (tahmini POS ile)
    clause = clause_crf.predict_single(feats)
    return tokens, pos, chunk, clause


def main():
    # Windows'ta çıktı dosyaya yönlendirilse de Türkçe karakterler bozulmasın
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = ("Dün akşam toplantıdan erken çıkan öğrencinin, hocasının önerdiği "
                "makaleyi kütüphanede dikkatlice okuduğunu fark ettim.")

    pos_crf = load("crf_pos.pkl")
    chunk_crf = load("crf_chunk.pkl")
    clause_crf = load("crf_clause.pkl")

    tokens, pos, chunk, clause = predict(text, pos_crf, chunk_crf, clause_crf)

    print("# text = %s" % text)
    print("# columns = ID FORM UPOS CHUNK CLAUSE")
    for i, (tk, p, ch, cl) in enumerate(zip(tokens, pos, chunk, clause), 1):
        print("%d\t%s\t%s\t%s\t%s" % (i, tk, p, ch, cl))


if __name__ == "__main__":
    main()
