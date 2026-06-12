# -*- coding: utf-8 -*-
"""
Proje raporunu tek dosyalık, taşınabilir bir HTML olarak üretir (report/RAPOR.html).
- Metrikler ve örnek çıktı modellerden CANLI hesaplanır.
- Grafikler (PNG) base64 olarak gömülür -> tek dosya, kolayca PDF'e çevrilir.
- Yazdırıldığında ~5 sayfa (A4) olacak şekilde CSS ile düzenlenmiştir.

Öğrenci bilgisi için aşağıdaki OGRENCILER listesini düzenleyin.
Kullanım:  python src/generate_report_html.py
"""
import os
import base64
import pickle

from sklearn.metrics import accuracy_score
from seqeval.metrics import (precision_score, recall_score, f1_score,
                             classification_report as seq_report)

from features import read_conll, sent2features, sent2labels
import predict as P

# ---- DÜZENLE ----
OGRENCILER = ["Mustafa Can Ersoy — 20360859046"]
# -----------------

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "chunks")
RESULTS = os.path.join(HERE, "..", "results")
REPORT = os.path.join(HERE, "..", "report")
MODELS = os.path.join(HERE, "..", "models")
ASSETS = os.path.join(HERE, "..", "assets")

ORNEK_CUMLE = ("Dün akşam toplantıdan erken çıkan öğrencinin, hocasının önerdiği "
               "makaleyi kütüphanede dikkatlice okuduğunu fark ettim.")


def load(f):
    with open(os.path.join(MODELS, f), "rb") as fh:
        return pickle.load(fh)


def img_b64_abs(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img_b64(fname):
    return img_b64_abs(os.path.join(RESULTS, fname))


def flat(seqs):
    return [x for s in seqs for x in s]


def eval_task(crf, test_sents, col, use_upos, pos_model=None):
    X, yt = [], []
    for s in test_sents:
        pp = None
        if use_upos and pos_model is not None:
            pp = pos_model.predict_single(sent2features(s, use_upos=False))
        X.append(sent2features(s, use_upos=use_upos, use_pred_pos=pp))
        yt.append(sent2labels(s, col))
    yp = crf.predict(X)
    acc = accuracy_score(flat(yt), flat(yp))
    out = {"acc": acc}
    if col in (2, 3):
        out["P"] = precision_score(yt, yp)
        out["R"] = recall_score(yt, yp)
        out["F1"] = f1_score(yt, yp)
        out["report"] = seq_report(yt, yp, output_dict=True, zero_division=0)
    return out


def pct(x):
    return "%.1f%%" % (100 * x)


def f3(x):
    return "%.3f" % x


def main():
    os.makedirs(REPORT, exist_ok=True)
    test = read_conll(os.path.join(DATA, "test.conll"))
    pos = load("crf_pos.pkl")
    chunk = load("crf_chunk.pkl")
    clause = load("crf_clause.pkl")

    r_pos = eval_task(pos, test, 1, False)
    r_chunk = eval_task(chunk, test, 2, True)
    r_e2e = eval_task(chunk, test, 2, True, pos_model=pos)
    r_clause = eval_task(clause, test, 3, True)

    # örnek çıktı (uçtan uca)
    toks, p_pos, p_chunk, p_clause = P.predict(ORNEK_CUMLE, pos, chunk, clause)

    # --- chunk sınıf tablosu satırları ---
    order = ["NP", "VP", "ADJP", "ADVP", "PP"]
    rep = r_chunk["report"]
    chunk_rows = ""
    for k in order:
        if k in rep:
            d = rep[k]
            chunk_rows += ("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                           "<td>%d</td></tr>") % (
                k, f3(d["precision"]), f3(d["recall"]), f3(d["f1-score"]),
                int(d["support"]))
    mic = rep.get("micro avg", {})
    chunk_rows += ("<tr class='tot'><td>Genel (micro)</td><td>%s</td><td>%s</td>"
                   "<td>%s</td><td>%d</td></tr>") % (
        f3(mic.get("precision", 0)), f3(mic.get("recall", 0)),
        f3(mic.get("f1-score", 0)), int(mic.get("support", 0)))

    # --- örnek çıktı satırları ---
    ex_rows = ""
    for i, (t, pp, pc, pl) in enumerate(zip(toks, p_pos, p_chunk, p_clause), 1):
        ex_rows += ("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td>"
                    "<td>%s</td></tr>") % (i, t, pp, pc, pl)

    html = TEMPLATE.format(
        ogrenciler="<br>".join(OGRENCILER),
        chunk_acc=pct(r_chunk["acc"]), chunk_f1=f3(r_chunk["F1"]),
        chunk_p=f3(r_chunk["P"]), chunk_r=f3(r_chunk["R"]),
        pos_acc=pct(r_pos["acc"]),
        e2e_acc=pct(r_e2e["acc"]), e2e_f1=f3(r_e2e["F1"]),
        clause_acc=pct(r_clause["acc"]), clause_f1=f3(r_clause["F1"]),
        chunk_rows=chunk_rows, ex_rows=ex_rows, ornek=ORNEK_CUMLE,
        cm_chunk=img_b64("confusion_chunk.png"),
        f1_chunk=img_b64("f1_chunk.png"),
        logo=img_b64_abs(os.path.join(ASSETS, "btu_logo.png")),
        n_train=3435, n_test=1100,
    )

    out = os.path.join(REPORT, "RAPOR.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("HTML rapor olusturuldu ->", os.path.abspath(out))


TEMPLATE = u"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>DDİ Projesi - Chunking Raporu</title>
<style>
  @page {{ size: A4; margin: 14mm 16mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; color:#1a1a1a;
         font-size: 10.3pt; line-height: 1.42; margin:0; }}
  h1 {{ font-size: 17pt; margin:0 0 2px; color:#16335c; text-align:center; }}
  .sub {{ text-align:center; font-size:11pt; color:#16335c; font-weight:600;
          margin-bottom:6px; }}
  .meta {{ text-align:center; font-size:9.5pt; color:#444; margin-bottom:4px; }}
  h2 {{ font-size:11.5pt; color:#16335c; border-bottom:2px solid #d7e0ee;
        padding-bottom:2px; margin:13px 0 5px; }}
  p {{ margin:4px 0; }}
  ul {{ margin:4px 0 4px 0; padding-left:18px; }}
  li {{ margin:1px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:9.3pt; margin:5px 0; }}
  th, td {{ border:1px solid #c4cdda; padding:2px 6px; text-align:center; }}
  th {{ background:#eaf0f8; color:#16335c; }}
  td:first-child, th:first-child {{ text-align:left; }}
  .tot td {{ font-weight:700; background:#f3f7fc; }}
  .kpi {{ display:flex; gap:8px; margin:6px 0; }}
  .kpi div {{ flex:1; border:1px solid #c4cdda; border-radius:6px; padding:6px;
              text-align:center; background:#f7faff; }}
  .kpi b {{ display:block; font-size:15pt; color:#16335c; }}
  .kpi span {{ font-size:8.6pt; color:#555; }}
  figure {{ margin:6px 0; text-align:center; page-break-inside:avoid; }}
  figure img {{ max-width:100%; border:1px solid #d7e0ee; }}
  figcaption {{ font-size:8.6pt; color:#555; margin-top:2px; }}
  code, .mono {{ font-family:Consolas,"Courier New",monospace; font-size:9pt; }}
  .note {{ font-size:8.8pt; color:#555; }}
  .row {{ display:flex; gap:10px; align-items:flex-start; }}
  .row > div {{ flex:1; }}
  .pb {{ page-break-before: always; }}

  /* ── Kapak ── */
  .cover {{ display:flex; flex-direction:column; align-items:center; text-align:center;
    min-height:252mm; padding:6mm 0 4mm; page-break-after:always;
    justify-content:space-between; font-family:"Times New Roman",Times,serif; color:#1a1a1a; }}
  .cover-univ {{ font-size:20pt; font-weight:bold; letter-spacing:.3px; margin-bottom:6px; }}
  .cover-fakulte {{ font-size:13.5pt; font-weight:bold; margin-bottom:4px; }}
  .cover-bolum {{ font-size:12pt; color:#333; }}
  .cover-logo-wrap {{ margin:20px 0; }}
  .cover-logo {{ height:150px; width:auto; display:block; margin:0 auto; }}
  .cover-title {{ font-size:17pt; font-weight:bold; line-height:1.45; margin-bottom:8px; }}
  .cover-subtitle {{ font-size:11.5pt; color:#444; font-style:italic; }}
  .cover-bottom {{ padding-top:8mm; }}
  .cover-ders {{ font-size:11.5pt; color:#555; margin-bottom:22px; }}
  .cover-danisman {{ font-size:12.5pt; font-weight:bold; margin-bottom:16px; }}
  .cover-ogrenci {{ font-size:12.5pt; font-weight:bold; margin-bottom:2px; }}
  .cover-no {{ font-size:11.5pt; color:#333; margin-bottom:16px; }}
  .cover-tarih {{ font-size:11.5pt; color:#555; }}
</style></head><body>

<div class="cover">
  <div>
    <div class="cover-univ">BURSA TEKNİK ÜNİVERSİTESİ</div>
    <div class="cover-fakulte">MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ</div>
    <div class="cover-bolum">Bilgisayar Mühendisliği Bölümü</div>
  </div>
  <div class="cover-logo-wrap">
    <img src="{logo}" class="cover-logo" alt="Bursa Teknik Üniversitesi">
  </div>
  <div>
    <div class="cover-title">İsim ve Öbeklerin Saptanması (Chunking)</div>
    <div class="cover-subtitle">Doğal Dil İşleme — Proje Konusu 3 · Türkçe için CRF Tabanlı Öbekleme</div>
  </div>
  <div class="cover-bottom">
    <div class="cover-ders">Doğal Dil İşleme — BLM0467</div>
    <div class="cover-danisman">Dr. Öğr. Üyesi HAYRİ VOLKAN AGUN</div>
    <div class="cover-ogrenci">Mustafa Can Ersoy</div>
    <div class="cover-no">20360859046</div>
    <div class="cover-tarih">Haziran 2026</div>
  </div>
</div>

<h1>Doğal Dil İşleme Projesi</h1>
<div class="sub">Konu 3: İsim ve Öbeklerin Saptanması (Chunking)</div>
<div class="meta">Bursa Teknik Üniversitesi · Bilgisayar Mühendisliği · 2025–2026 Bahar</div>

<div class="kpi">
  <div><b>{chunk_acc}</b><span>Chunk - Kelime Doğruluğu</span></div>
  <div><b>{chunk_f1}</b><span>Chunk - Öbek F1</span></div>
  <div><b>{pos_acc}</b><span>POS Doğruluğu</span></div>
</div>

<h2>1. Ne Yaptık?</h2>
<p>Verilen bir Türkçe cümlede yer alan <b>öbekleri</b> (isim NP, eylem VP, sıfat
ADJP, zarf ADVP, edat PP) ve cümle içindeki <b>yan cümlecikleri</b> (ilgi RELCL,
tümleç COMPCL, zarf ADVCL) otomatik olarak işaretleyen bir <b>chunking (sığ
ayrıştırma)</b> sistemi geliştirdik. Problemi <b>dizisel etiketleme (sequence
labeling)</b> olarak modelledik ve istatistiksel bir makine öğrenmesi yöntemi olan
<b>CRF (Conditional Random Fields)</b> ile çözdük. Tüm işaretlemeler
<b>CoNLL / BIO</b> formatındadır (B = öbek başı, I = devamı, O = öbek dışı).
Sistem, ham bir cümleyi alıp uçtan uca <b>POS → CHUNK → CLAUSE</b> olarak
etiketleyebilmektedir.</p>

<h2>2. Neler Kullandık?</h2>
<div class="row">
<div>
<ul>
<li><b>Veri:</b> Universal Dependencies <b>Türkçe-IMST</b> treebank (CC BY-SA).
Bağımlılık ağacından kurallarla öbek/cümlecik BIO etiketleri türetildi.</li>
<li><b>Model:</b> CRF — <code>sklearn-crfsuite</code>, L-BFGS, L1/L2
düzenlileştirme.</li>
<li><b>Kütüphaneler:</b> scikit-learn, sklearn-crfsuite, seqeval, matplotlib,
numpy.</li>
</ul>
</div>
<div>
<ul>
<li><b>Öznitelikler:</b> kelime, 1–4 harf ön/son ekler, kelime şekli, büyük/küçük
harf bayrakları, <b>UPOS</b> ve ±2 komşuluk bağlamı.</li>
<li><b>Veri büyüklüğü:</b> {n_train} eğitim / {n_test} test cümlesi.</li>
<li><b>3 model:</b> POS, CHUNK (ana görev), CLAUSE.</li>
</ul>
</div>
</div>

<p class="note"><b>Örnek CoNLL/BIO işaretleme:</b>
<span class="mono">1 Dün ADV B-ADVP &nbsp; 3 toplantıdan NOUN B-NP &nbsp;
5 çıkan VERB B-VP &nbsp; 16 . PUNCT O</span></p>

<h2>3. Sonuçlar — Ana Görev: Chunking</h2>
<p>Test kümesinde sınıf bazında başarı (öbek/entity düzeyi, seqeval):</p>
<table>
<tr><th>Öbek</th><th>Precision</th><th>Recall</th><th>F1</th><th>Adet</th></tr>
{chunk_rows}
</table>
<p class="note">Kelime düzeyi doğruluk: <b>{chunk_acc}</b> ·
Genel öbek F1: <b>{chunk_f1}</b> (P={chunk_p}, R={chunk_r}).</p>

<div class="row">
<figure><img src="{cm_chunk}">
<figcaption>Karışıklık matrisi (her sınıf, normalize) — Chunking</figcaption></figure>
</div>
<figure><img src="{f1_chunk}" style="max-width:88%">
<figcaption>Sınıf bazında F1 — Chunking</figcaption></figure>

<h2>4. Diğer Sonuçlar ve Örnek Çıktı</h2>
<table>
<tr><th>Görev</th><th>Kelime Doğruluğu</th><th>Öbek F1</th></tr>
<tr><td>POS etiketleme</td><td>{pos_acc}</td><td>—</td></tr>
<tr><td>Chunking (gold POS)</td><td>{chunk_acc}</td><td>{chunk_f1}</td></tr>
<tr><td>Chunking (uçtan uca, tahmini POS)</td><td>{e2e_acc}</td><td>{e2e_f1}</td></tr>
<tr><td>Clause (yan cümlecik)</td><td>{clause_acc}</td><td>{clause_f1}</td></tr>
</table>
<p class="note">Uçtan uca senaryoda POS da model tarafından tahmin edildiğinden
başarı bir miktar düşer; bu, POS bilgisinin chunking için değerini gösterir.
Cümlecik katmanı uzun ve iç içe kapsamlar içerdiğinden öbek-düzeyi F1'i düşüktür,
kelime düzeyi doğruluğu yüksektir.</p>

<p><b>Örnek çıktı</b> (ham cümle → sistem işaretlemesi):<br>
<span class="note mono">{ornek}</span></p>
<table>
<tr><th>#</th><th>Kelime</th><th>POS</th><th>CHUNK</th><th>CLAUSE</th></tr>
{ex_rows}
</table>

<p class="note">Çalıştırma: <code>python src/run_all.py</code> (indir→eğit→test→rapor)
veya <code>python src/predict.py "cümleniz"</code>. Kaynak kod <code>src/</code>,
veri <code>data/chunks/</code>, eğitilmiş modeller <code>models/</code>,
metrik ve grafikler <code>results/</code> klasöründedir.</p>

</body></html>
"""

if __name__ == "__main__":
    main()
