# -*- coding: utf-8 -*-
"""
Tüm boru hattını tek komutla çalıştırır:
  1) UD verisini indir   2) Chunk CoNLL'e dönüştür
  3) Modelleri eğit      4) Değerlendir (+grafikler)
  5) Uçtan uca değerlendir

Kullanım:  python src/run_all.py
"""
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))


def step(title, module, argv=None):
    print("\n" + "#" * 64)
    print("# " + title)
    print("#" * 64)
    sys.argv = [module] + (argv or [])
    runpy.run_path(os.path.join(HERE, module), run_name="__main__")


def main():
    ud = os.path.join(HERE, "..", "data", "ud", "tr_imst-ud-train.conllu")
    if not os.path.exists(ud):
        step("1/5  Veri indiriliyor", "download_data.py")
    else:
        print("1/5  Veri zaten mevcut, indirme atlandı.")
    step("2/5  CoNLL-U -> Chunk CoNLL dönüşümü", "conllu_to_chunks.py")
    step("3/5  Modeller eğitiliyor", "train.py")
    step("4/5  Değerlendirme (gold POS)", "evaluate.py")
    step("5/5  Uçtan uca değerlendirme (predicted POS)", "evaluate.py",
         ["--task", "chunk", "--end2end"])
    print("\nTAMAMLANDI. Çıktılar: data/chunks, models, results")


if __name__ == "__main__":
    main()
