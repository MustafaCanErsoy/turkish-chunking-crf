# -*- coding: utf-8 -*-
"""
Proje raporunu tek dosyalık, taşınabilir bir HTML olarak üretir (kök dizinde RAPOR.html).
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

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "chunks")
RESULTS = os.path.join(HERE, "..", "results")
REPORT = os.path.join(HERE, "..")   # rapor proje kök dizinine yazılır
MODELS = os.path.join(HERE, "..", "models")
ASSETS = os.path.join(HERE, "..", "assets")

ORNEK_CUMLE = "Öğrenci kütüphanede yeni kitabı okudu."


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
  @page {{ size: A4; margin: 15mm 18mm 13mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Times New Roman", Times, serif; color:#1a1a1a;
         font-size: 10.7pt; line-height: 1.4; margin:0; }}
  h1 {{ font-size: 15.5pt; margin:0 0 2px; color:#1a1a1a; text-align:center; }}
  .sub {{ text-align:center; font-size:11pt; color:#333; font-weight:bold;
          margin-bottom:2px; }}
  .meta {{ text-align:center; font-size:9.5pt; color:#555; margin-bottom:9px;
           padding-bottom:6px; border-bottom:1.5px solid #333; }}
  h2 {{ font-size:12.5pt; color:#2c3e50; margin:12px 0 5px; padding-bottom:3px;
        border-bottom:1.5px solid #d0d7df; page-break-after:avoid; }}
  h3 {{ font-size:11pt; color:#34495e; margin:8px 0 3px; page-break-after:avoid; }}
  p {{ margin:4px 0; text-align:justify; }}
  ul {{ margin:4px 0 6px 20px; padding:0; }}
  li {{ margin:1px 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:9.7pt; margin:6px 0;
           page-break-inside:avoid; }}
  table.veri th {{ background:#2c3e50; color:#fff; padding:5px 9px; text-align:center; }}
  table.veri td {{ padding:4px 9px; border-bottom:1px solid #e0e0e0; text-align:center; }}
  table.veri tr:nth-child(even) td {{ background:#f7f9fc; }}
  table.veri td:first-child {{ text-align:left; }}
  .tot td {{ font-weight:bold; background:#eef3f9; color:#1a6830; }}
  figure {{ margin:7px auto; text-align:center; page-break-inside:avoid; }}
  figure img {{ max-width:100%; border:1px solid #dce3ea; }}
  figcaption {{ font-size:9pt; color:#555; margin-top:3px; font-style:italic; }}
  code, .mono {{ font-family:"Courier New",monospace; font-size:9.3pt; }}
  .note {{ font-size:9pt; color:#444; }}
  .pb {{ page-break-before: always; }}
  .abstract {{ background:#f7f9fc; border-left:4px solid #2980b9; padding:9px 14px;
    margin:4px 0 10px; font-size:10pt; line-height:1.4; text-align:justify;
    page-break-inside:avoid; }}
  .abstract h2 {{ font-size:11.5pt; margin:0 0 4px; border:none; padding:0; color:#2c3e50; }}
  .abstract a {{ color:#2061a8; }}
  .kv {{ margin-top:6px; }}

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

<h1>İsim ve Öbeklerin Saptanması (Chunking)</h1>
<div class="sub">Doğal Dil İşleme — Proje Konusu 3</div>
<div class="meta">Bursa Teknik Üniversitesi · Bilgisayar Mühendisliği Bölümü · 2025–2026 Bahar</div>

<div class="abstract">
  <h2>Özet</h2>
  <b>Ne yaptık?</b> Verilen bir Türkçe cümlede yer alan öbekleri (isim NP, eylem VP,
  sıfat ADJP, zarf ADVP, edat PP) ve yan cümlecikleri (RELCL, COMPCL, ADVCL)
  otomatik olarak işaretleyen, dizisel etiketleme temelli bir <i>chunking (sığ
  ayrıştırma)</i> sistemi geliştirdik. Tüm işaretlemeler CoNLL / BIO biçimindedir.
  <b>Neler kullandık?</b> Veri Universal Dependencies Türkçe-IMST treebank'inden
  türetilmiş; model olarak istatistiksel <i>CRF (Conditional Random Fields)</i>
  kullanılmış; öznitelik çıkarımı ve değerlendirme Python (scikit-learn,
  sklearn-crfsuite, seqeval, matplotlib) ile yapılmıştır. Sistem ana görevde
  <b>kelime düzeyinde {chunk_acc} doğruluk</b> ve <b>öbek düzeyinde {chunk_f1} F1</b>
  elde etmiştir.
  <div class="kv"><b>Anahtar Kelimeler:</b> Chunking, CRF, Dizisel Etiketleme,
  CoNLL/BIO, Türkçe Doğal Dil İşleme. &nbsp;|&nbsp; <b>GitHub:</b>
  <a href="https://github.com/MustafaCanErsoy/turkish-chunking-crf">github.com/MustafaCanErsoy/turkish-chunking-crf</a></div>
</div>

<h2>1. Problem Tanımı</h2>
<p>Bir cümle, tek tek kelimelerden değil, anlamlı kelime gruplarından oluşur: bir
isim ile onu niteleyen sözcükler bir isim öbeği, bir eylem ile yardımcıları bir
eylem öbeği oluşturur. Chunking (sığ ayrıştırma), cümleyi tam bir sözdizimsel ağaca
ayrıştırmadan bu temel öbekleri bulup sınıflandırma görevidir. Tam ayrıştırmaya göre
daha hızlı ve hatalara karşı daha dayanıklı olduğundan; bilgi çıkarımı, varlık
tanıma ve makine çevirisi gibi uygulamalarda sık kullanılan bir ön işleme adımıdır.</p>
<p>Bu çalışmada iki katmanlı bir işaretleme yapılmıştır. Birincil katman olan
<b>CHUNK</b>, beş temel öbeği ayırt eder: isim öbeği (NP), eylem öbeği (VP), sıfat
öbeği (ADJP), zarf öbeği (ADVP) ve edat öbeği (PP). İkincil katman olan <b>CLAUSE</b>
ise cümle içindeki yan cümlecikleri kapsar: ilgi cümleciği (RELCL), tümleç cümleciği
(COMPCL) ve zarf cümleciği (ADVCL).</p>
<p>Her iki katman da <b>BIO şeması</b> ile kodlanır. Bu şemada bir öbeğin ilk
kelimesi <code>B-</code> (başlangıç), sonraki kelimeleri <code>I-</code> (iç) ve
hiçbir öbeğe girmeyen kelimeler <code>O</code> (dışında) etiketini alır; böylece öbek
sınırları kelime dizisi üzerinde belirsizliğe yer bırakmadan gösterilebilir. Örneğin
&ldquo;Dün toplantıdan çıkan öğrenci&rdquo; parçasında &ldquo;Dün&rdquo; tek başına
bir zarf öbeğidir (B-ADVP), &ldquo;toplantıdan çıkan öğrenci&rdquo; ise tek bir isim
öbeği oluşturur (B-NP, I-NP, I-NP). Bu kodlama sayesinde problem, her kelimeye bir
etiket atanan bir <b>dizisel etiketleme (sequence labeling)</b> problemine dönüşür.</p>

<h2>2. Kullanılan Yöntemler ve Araçlar</h2>
<h3>2.1 Veri ve İşaretleme</h3>
<p>Türkçe için elle işaretlenmiş hazır bir chunking veri kümesi bulunmadığından,
eğitim ve test verisi açık lisanslı Universal Dependencies <b>Türkçe-IMST</b>
treebank'inden türetilmiştir. Bu treebank her cümle için sözcük türlerini (POS) ve
kelimeler arasındaki bağımlılık ilişkilerini içerir. Bağımlılık ağacı üzerinde
tanımladığımız dilbilgisi kurallarıyla, bir baş kelime ile ona bağlı sözcüklerin
oluşturduğu öbekler ve cümleciklerin kapsamları otomatik olarak BIO etiketlerine
çevrilmiştir. Elde edilen veri toplam <b>{n_train} eğitim</b> ve <b>{n_test} test</b>
cümlesi içermekte ve CoNLL biçiminde proje dosyasında yer almaktadır.</p>
<h3>2.2 Öznitelikler</h3>
<p>Modelin her kelime için kullandığı öznitelikler yüzeysel ve bağlamsaldır:
kelimenin kendisi, baştaki ve sondaki 1–4 harflik ekleri, büyük/küçük harf deseni,
POS etiketi ve hemen çevresindeki iki komşuya kadar olan kelimelerin sözcük ve POS
bilgileri. Türkçe sondan eklemeli bir dil olduğu için son ekler öbek türü hakkında
güçlü ipuçları taşır; örneğin &ldquo;-da/-de&rdquo; eki çoğu zaman bir ismin, dolaylı
olarak bir isim öbeğinin sonunu, fiil çekim ekleri ise bir eylem öbeğini işaret
eder. Bu nedenle ek temelli öznitelikler başarıda belirleyici rol oynar.</p>
<h3>2.3 Model</h3>
<p>Sınıflandırıcı olarak <b>CRF (Conditional Random Fields)</b> seçilmiştir. CRF, her
kelimeyi tek tek sınıflandırmak yerine cümlenin tüm etiket dizisini birlikte
değerlendirir ve etiketler arasındaki geçiş kurallarını da öğrenir. Böylece
&ldquo;I-NP yalnızca B-NP ya da I-NP'den sonra gelebilir&rdquo; gibi kısıtları
kendiliğinden uygulayarak geçersiz etiket dizilerini önler; bu da onu BIO tabanlı
öbekleme için doğal ve güçlü bir seçim yapar. Model, <code>sklearn-crfsuite</code>
kütüphanesiyle L-BFGS optimizasyonu ve L1/L2 düzenlileştirme kullanılarak
eğitilmiştir. Toplam üç model eğitilmiştir: ham metni işaretleyebilmek için bir POS
etiketleyici, ana görev için CHUNK modeli ve yan cümlecikler için CLAUSE modeli.
Değerlendirme ve grafikler scikit-learn, seqeval ve matplotlib ile üretilmiştir.</p>

<h2>3. Sonuçlar (Çıktılar)</h2>
<p>Ana görev olan chunking için test kümesinde sınıf bazında öbek (entity) düzeyi
başarı değerleri Tablo 1'de verilmiştir.</p>
<table class="veri">
<tr><th>Öbek</th><th>Precision</th><th>Recall</th><th>F1</th><th>Adet</th></tr>
{chunk_rows}
</table>
<p class="note">Tablo 1. Chunking sınıf bazında başarı. Kelime düzeyi doğruluk:
<b>{chunk_acc}</b>; genel öbek F1: <b>{chunk_f1}</b> (P={chunk_p}, R={chunk_r}).</p>

<figure><img src="{cm_chunk}" style="max-width:72%">
<figcaption>Şekil 1. Karışıklık matrisi (her sınıf, normalize) — Chunking.</figcaption></figure>
<figure><img src="{f1_chunk}" style="max-width:84%">
<figcaption>Şekil 2. Sınıf bazında F1 değerleri — Chunking.</figcaption></figure>

<p>Sonuçlar incelendiğinde, en yüksek başarı sınırları belirgin olan zarf ve isim
öbeklerinde elde edilmiştir; zarf öbekleri çoğunlukla tek kelimeden oluştuğu için
neredeyse kusursuz ayrılır. Buna karşılık sıfat ve edat öbekleri daha zordur:
sıfatlar çoğu zaman bir isim öbeğinin içinde yer aldığından model onları sık sık isim
öbeğiyle karıştırır. Bu durum Şekil 1'deki karışıklık matrisinde ADJP ile NP
arasındaki karışma olarak açıkça görülmektedir. Edat öbeklerinin görece az sayıda
örnekle temsil edilmesi de bu sınıftaki başarıyı sınırlamaktadır.</p>
<p>Tablo 2, tüm görevlerin özetini sunar. Gerçekçi &ldquo;uçtan uca&rdquo; senaryoda,
modele hazır POS etiketi verilmez ve POS da sistemin kendisi tarafından tahmin
edilir; bu durumda öbek F1'i {chunk_f1}'den {e2e_f1}'e düşer. Bu düşüş, doğru POS
bilgisinin öbekleme için ne kadar değerli olduğunu somut biçimde ortaya koyar. Yan
cümlecik (CLAUSE) katmanı ise en zorlu görevdir: cümlecikler uzun ve çoğu zaman iç
içe geçen kapsamlar oluşturduğundan, tam kapsam eşleşmesi gerektiren öbek F1'i düşük
kalır; yine de kelime düzeyindeki doğruluk ({clause_acc}) makul bir seviyededir.</p>
<table class="veri">
<tr><th>Görev</th><th>Kelime Doğruluğu</th><th>Öbek F1</th></tr>
<tr><td>POS etiketleme</td><td>{pos_acc}</td><td>—</td></tr>
<tr><td>Chunking (gold POS)</td><td>{chunk_acc}</td><td>{chunk_f1}</td></tr>
<tr><td>Chunking (uçtan uca, tahmini POS)</td><td>{e2e_acc}</td><td>{e2e_f1}</td></tr>
<tr><td>Clause (yan cümlecik)</td><td>{clause_acc}</td><td>{clause_f1}</td></tr>
</table>
<p class="note">Tablo 2. Tüm görevlerin özet başarı karşılaştırması.</p>

<h2>4. Örnek Çıktı ve Sonuç</h2>
<p>Eğitilmiş sistemin ham bir cümleye ürettiği uçtan uca çıktı (<i>{ornek}</i>):</p>
<table class="veri">
<tr><th>#</th><th>Kelime</th><th>POS</th><th>CHUNK</th><th>CLAUSE</th></tr>
{ex_rows}
</table>
<p>Sonuç olarak, Türkçe için CRF tabanlı bir öbekleme sistemi geliştirilmiş ve eğitim
ile test süreçlerini uçtan uca yürüten çalışır bir boru hattı kurulmuştur. Ana
görevde kelime düzeyinde {chunk_acc} doğruluk ve öbek düzeyinde {chunk_f1} F1 elde
edilmiş; tüm işaretlemeler CoNLL/BIO biçiminde üretilmiş ve her sınıf için başarı
oranları ile karışıklık matrisi raporlanmıştır. Çalışmanın başlıca sınırlılığı,
etiketlerin bir treebank'ten kurallarla türetilmiş olmasıdır. İleride elle
işaretlenmiş veri, morfolojik öznitelikler (örneğin Zemberek çözümlemeleri) veya
kelime vektörleri eklenerek ve özellikle sıfat/edat öbekleri ile cümlecik katmanına
odaklanılarak başarının daha da artırılması mümkündür.</p>

</body></html>
"""

if __name__ == "__main__":
    main()
