# -*- coding: utf-8 -*-
"""
UD CoNLL-U -> Chunk CoNLL donusturucu.

Bağımlılık (dependency) ağacından deterministik kurallarla iki etiket katmanı
üretir ve PDF'teki Chunking formatında (BIO) yazar:

  CHUNK katmanı  : Temel öbekler  -> NP, VP, ADJP, ADVP, PP   (CoNLL-2000 tarzı)
  CLAUSE katmanı : Yan cümlecikler -> RELCL, COMPCL, ADVCL

Çıktı sütunları:  ID  FORM  UPOS  CHUNK  CLAUSE

Kurallar kısaca
---------------
CHUNK: Her kelime, yalnızca "öbek-içi" (intra-phrase) bağıntılarla ulaşılan en
       üst başına (head) bağlanır. Aynı başa bağlı ve bitişik kelimeler tek bir
       öbek olur. Öbeğin türü, baş kelimenin UPOS etiketinden belirlenir.
CLAUSE: acl -> RELCL, ccomp/csubj/xcomp -> COMPCL, advcl -> ADVCL. İlgili
        kelimenin tüm alt ağacı (subtree) o cümleciğin kapsamı olarak işaretlenir.
"""
import os
import io
import argparse

# Öbek-içi (intra-phrase) bağıntılar: bunlarla bağlı kelime başıyla aynı öbekte kalır.
INTRA = {
    "det", "amod", "nummod", "nmod:poss", "compound", "compound:lvc",
    "compound:redup", "flat", "flat:name", "case", "fixed", "goeswith",
    "cop", "aux", "aux:q", "det:predet", "advmod:emph", "clf",
}

# UPOS -> öbek türü
UPOS2CHUNK = {
    "NOUN": "NP", "PROPN": "NP", "PRON": "NP", "NUM": "NP", "DET": "NP",
    "VERB": "VP", "AUX": "VP",
    "ADJ": "ADJP",
    "ADV": "ADVP",
    "ADP": "PP",
}
# Bunlar her zaman tek başına O (öbek dışı)
O_UPOS = {"PUNCT", "CCONJ", "SCONJ", "INTJ", "SYM", "X", "PART"}

# Clause türü eşlemesi (deprel -> clause etiketi)
CLAUSE_MAP = {
    "acl": "RELCL", "acl:relcl": "RELCL",
    "ccomp": "COMPCL", "csubj": "COMPCL", "csubj:pass": "COMPCL", "xcomp": "COMPCL",
    "advcl": "ADVCL",
}


def read_conllu(path):
    """CoNLL-U dosyasını cümle listesi olarak okur. Her cümle: (meta_text, tokens).
    tokens: list of dict(id, form, upos, head, deprel)."""
    sentences = []
    with io.open(path, encoding="utf-8") as f:
        toks, text = [], None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                if line.startswith("# text ="):
                    text = line.split("=", 1)[1].strip()
                continue
            if line == "":
                if toks:
                    sentences.append((text, toks))
                toks, text = [], None
                continue
            cols = line.split("\t")
            tid = cols[0]
            if "-" in tid or "." in tid:   # çok-kelimeli token / boş düğüm -> atla
                continue
            toks.append({
                "id": int(tid),
                "form": cols[1],
                "upos": cols[3],
                "head": int(cols[6]) if cols[6] != "_" else 0,
                "deprel": cols[7].split(":")[0] if cols[7] != "_" else "_",
                "deprel_full": cols[7],
            })
        if toks:
            sentences.append((text, toks))
    return sentences


def build_index(toks):
    by_id = {t["id"]: t for t in toks}
    children = {t["id"]: [] for t in toks}
    children[0] = []
    for t in toks:
        children.setdefault(t["head"], []).append(t["id"])
    return by_id, children


def subtree_span(root_id, children):
    """Bir kelimenin alt ağacındaki min..max kelime indekslerini döndürür."""
    stack, ids = [root_id], []
    while stack:
        cur = stack.pop()
        ids.append(cur)
        stack.extend(children.get(cur, []))
    return min(ids), max(ids), set(ids)


def chunk_root(tid, by_id):
    """Yalnızca öbek-içi bağıntılarla ulaşılan en üst başı bulur."""
    cur = tid
    seen = set()
    while True:
        t = by_id[cur]
        if t["deprel"] in INTRA and t["head"] != 0 and t["head"] in by_id and cur not in seen:
            seen.add(cur)
            cur = t["head"]
        else:
            return cur


def assign_chunks(toks, by_id):
    """Her kelimeye CHUNK etiketi (B-/I-/O) atar."""
    # 1) her kelimenin öbek-kökü
    roots = {t["id"]: chunk_root(t["id"], by_id) for t in toks}
    labels = ["O"] * len(toks)
    i = 0
    n = len(toks)
    while i < n:
        head_id = roots[toks[i]["id"]]
        head_upos = by_id[head_id]["upos"]
        # öbek dışı kelimeler (noktalama, bağlaç...) tek başına O
        if head_upos in O_UPOS:
            labels[i] = "O"
            i += 1
            continue
        # aynı köke sahip bitişik kelimeleri topla
        j = i
        members = []
        while j < n and roots[toks[j]["id"]] == head_id:
            members.append(j)
            j += 1
        ctype = UPOS2CHUNK.get(head_upos, "NP")
        # son kelime bir edat (postposition) ise öbeği PP yap (ör. "okul için")
        if by_id[toks[members[-1]]["id"]]["upos"] == "ADP" and ctype == "NP":
            ctype = "PP"
        for k, idx in enumerate(members):
            labels[idx] = ("B-" if k == 0 else "I-") + ctype
        i = j
    return labels


def assign_clauses(toks, by_id, children):
    """Her kelimeye CLAUSE etiketi (B-/I-/O) atar. İç içe cümleciklerde en küçük
    (en derin) kapsayan cümlecik kazanır."""
    n = len(toks)
    id2pos = {t["id"]: p for p, t in enumerate(toks)}
    # her clause başı için (span_size, type, pozisyon kümesi)
    spans = []
    for t in toks:
        ctype = CLAUSE_MAP.get(t["deprel_full"]) or CLAUSE_MAP.get(t["deprel"])
        if ctype:
            lo, hi, ids = subtree_span(t["id"], children)
            positions = sorted(id2pos[i] for i in ids if i in id2pos)
            spans.append((len(positions), ctype, positions))
    # küçük span'ler önce uygulanmasın diye: büyükten küçüğe yaz, küçük üzerine yazar
    spans.sort(key=lambda x: -x[0])
    labels = ["O"] * n
    span_of = [None] * n
    for size, ctype, positions in spans:
        for p in positions:
            span_of[p] = ctype
    # BIO ata: aynı clause türünün bitişik bloğunda ilk B-, devamı I-
    prev = None
    for p in range(n):
        c = span_of[p]
        if c is None:
            labels[p] = "O"
            prev = None
        else:
            labels[p] = ("B-" if c != prev else "I-") + c
            prev = c
    return labels


def convert_file(in_path, out_path):
    sents = read_conllu(in_path)
    n_sent = n_tok = 0
    with io.open(out_path, "w", encoding="utf-8") as out:
        for text, toks in sents:
            if not toks:
                continue
            by_id, children = build_index(toks)
            chunks = assign_chunks(toks, by_id)
            clauses = assign_clauses(toks, by_id, children)
            if text:
                out.write(u"# text = %s\n" % text)
            out.write(u"# columns = ID FORM UPOS CHUNK CLAUSE\n")
            for t, ch, cl in zip(toks, chunks, clauses):
                out.write(u"%d\t%s\t%s\t%s\t%s\n" % (t["id"], t["form"], t["upos"], ch, cl))
            out.write(u"\n")
            n_sent += 1
            n_tok += len(toks)
    print("  %-40s -> %-40s  (%d cümle, %d kelime)" % (
        os.path.basename(in_path), os.path.basename(out_path), n_sent, n_tok))


def main():
    here = os.path.dirname(__file__)
    ud = os.path.join(here, "..", "data", "ud")
    out = os.path.join(here, "..", "data", "chunks")
    os.makedirs(out, exist_ok=True)
    for split in ["train", "dev", "test"]:
        convert_file(
            os.path.join(ud, "tr_imst-ud-%s.conllu" % split),
            os.path.join(out, "%s.conll" % split),
        )
    print("Donusturme tamam ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
