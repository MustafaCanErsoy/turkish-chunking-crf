<h1 align="center">🧩 Türkçe Chunking — İsim ve Öbeklerin Saptanması</h1>

<p align="center">
  <b>Doğal Dil İşleme Projesi · Konu 3</b><br>
  Bursa Teknik Üniversitesi · Bilgisayar Mühendisliği · 2025–2026 Bahar
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Model" src="https://img.shields.io/badge/Model-CRF%20(sklearn--crfsuite)-4C72B0">
  <img alt="Format" src="https://img.shields.io/badge/Format-CoNLL%20%2F%20BIO-success">
  <img alt="Chunk F1" src="https://img.shields.io/badge/Chunk%20F1-0.838-brightgreen">
  <img alt="Accuracy" src="https://img.shields.io/badge/Token%20Acc-88.6%25-brightgreen">
</p>

---

Verilen bir Türkçe cümlede **isim (NP), eylem (VP), sıfat (ADJP), zarf (ADVP) ve
edat (PP) öbeklerini** ve cümle içindeki **yan cümlecikleri (RELCL / COMPCL / ADVCL)**
otomatik olarak işaretleyen, **CRF (Conditional Random Fields)** tabanlı bir
*chunking (sığ ayrıştırma)* sistemi. Tüm etiketleme **CoNLL / BIO** formatındadır.

## 📋 İçindekiler
- [Özellikler](#-özellikler)
- [Hızlı Başlangıç](#-hızlı-başlangıç)
- [Sonuçlar](#-sonuçlar)
- [Nasıl Çalışır?](#-nasıl-çalışır)
- [Proje Yapısı](#-proje-yapısı)
- [Örnek Çıktı](#-örnek-çıktı)

## ✨ Özellikler
- 🔤 **5 öbek türü** (NP, VP, ADJP, ADVP, PP) + **3 cümlecik türü** (RELCL, COMPCL, ADVCL)
- 🧠 **İstatistiksel ML**: Conditional Random Fields (dizisel etiketleme)
- 🔁 **Uçtan uca**: ham cümle → POS → CHUNK → CLAUSE
- 📊 Otomatik **precision / recall / F1 / accuracy + karışıklık matrisi + grafikler**
- 🇹🇷 Gerçek veri: **Universal Dependencies Türkçe-IMST** treebank

## 🚀 Hızlı Başlangıç

```bash
# 1) Bağımlılıklar
pip install -r requirements.txt

# 2) Tüm boru hattı: indir → dönüştür → eğit → test
python src/run_all.py

# 3) Ham bir cümleyi öbeklere ayır
python src/predict.py "Dün akşam toplantıdan erken çıkan öğrenci eve gitti."
```

> Windows kullanıcıları tek tıkla **`calistir.bat`** dosyasını çalıştırabilir.

## 📊 Sonuçlar

Test kümesi (1.100 cümle / 10.032 kelime):

| Görev | Kelime Doğruluğu | Öbek F1 |
|------|:---:|:---:|
| **Chunking** (gold POS) | **88.6%** | **0.838** |
| Chunking (uçtan uca, tahmini POS) | 82.4% | 0.756 |
| POS etiketleme | 91.2% | — |
| Clause (yan cümlecik) | 78.6% | 0.278 |

<p align="center">
  <img src="results/confusion_chunk.png" width="46%" alt="Karışıklık matrisi">
  <img src="results/f1_chunk.png" width="52%" alt="Sınıf bazında F1">
</p>

## 🔍 Nasıl Çalışır?

```
UD Türkçe-IMST (CoNLL-U)
        │  conllu_to_chunks.py  (bağımlılık ağacı → BIO etiket)
        ▼
data/chunks/*.conll  (ID FORM UPOS CHUNK CLAUSE)
        │  features.py  (kelime, ön/son ek, şekil, UPOS, ±2 bağlam)
        ▼
   CRF eğitimi (train.py)  →  models/*.pkl
        │
        ▼
   Değerlendirme (evaluate.py) → results/  ·  Tahmin (predict.py)
```

- **Veri:** UD Türkçe-IMST treebank'inin bağımlılık çözümlemesinden, deterministik
  dilbilgisi kurallarıyla öbek/cümlecik BIO etiketleri türetilir.
- **Öznitelikler:** küçük harf, 1–4 harf ön/son ekler, kelime şekli, büyük/küçük
  harf bayrakları, UPOS ve ±2 komşuluk bağlamı.
- **Model:** CRF (`sklearn-crfsuite`), L-BFGS, L1/L2 düzenlileştirme.

## 📁 Proje Yapısı

```
.
├── src/
│   ├── download_data.py        # UD treebank indir
│   ├── conllu_to_chunks.py     # CoNLL-U → Chunk CoNLL (BIO)
│   ├── features.py             # öznitelik çıkarımı
│   ├── train.py                # CRF eğitimi (pos, chunk, clause)
│   ├── evaluate.py             # metrik + confusion matrix + grafik
│   ├── predict.py              # ham cümle → öbekler (demo)
│   └── run_all.py              # uçtan uca boru hattı
├── data/
│   ├── ud/                     # ham UD treebank (CoNLL-U)
│   └── chunks/                 # etiketli veri (CoNLL/BIO)
├── models/                     # eğitilmiş CRF modelleri (.pkl)
├── results/                    # metrikler (.txt) + grafikler (.png)
├── RAPOR_20360859046_MustafaCanErsoy.pdf  # 📄 RAPOR (proje raporu, PDF)
├── requirements.txt
└── calistir.bat                # Windows tek-tık çalıştırma
```

## 🧪 Örnek Çıktı

```
$ python src/predict.py "Dün akşam toplantıdan erken çıkan öğrenci eve gitti."

# columns = ID FORM UPOS CHUNK CLAUSE
1  Dün          NOUN   B-NP    B-RELCL
2  akşam        NOUN   B-NP    I-RELCL
3  toplantıdan  NOUN   B-NP    I-RELCL
4  erken        ADV    B-ADVP  I-RELCL
5  çıkan        VERB   B-VP    I-RELCL
6  öğrenci      ADJ    B-NP    O
7  eve          NOUN   I-NP    O
8  gitti        VERB   B-VP    O
9  .            PUNCT  O       O
```

## 🏷️ Etiket Kümeleri
- **CHUNK:** `B-/I-` ile **NP, VP, ADJP, ADVP, PP** ve `O`
- **CLAUSE:** `B-/I-` ile **RELCL, COMPCL, ADVCL** ve `O`

## 📜 Lisans / Veri
Eğitim verisi [Universal Dependencies Türkçe-IMST](https://github.com/UniversalDependencies/UD_Turkish-IMST)
treebank'inden türetilmiştir (CC BY-SA 4.0).

---
<p align="center"><sub>Mustafa Can Ersoy — 20360859046</sub></p>
