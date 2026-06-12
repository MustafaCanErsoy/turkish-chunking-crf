# -*- coding: utf-8 -*-
"""
UD_Turkish-IMST treebank (CoNLL-U) dosyalarini indirir.
Kaynak: Universal Dependencies (CC BY-SA 4.0).
Cikti: data/ud/tr_imst-ud-{train,dev,test}.conllu
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-IMST/master/"
FILES = [
    "tr_imst-ud-train.conllu",
    "tr_imst-ud-dev.conllu",
    "tr_imst-ud-test.conllu",
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ud")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in FILES:
        url = BASE + fname
        out = os.path.join(OUT_DIR, fname)
        print(f"Indiriliyor: {fname} ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, out)
        size = os.path.getsize(out)
        print(f"OK ({size:,} bytes)")
    print("Tum dosyalar indirildi ->", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
