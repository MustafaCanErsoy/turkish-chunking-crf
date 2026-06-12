# -*- coding: utf-8 -*-
"""
CRF için özellik (feature) çıkarımı ve veri okuma yardımcıları.

Her kelime, kendisi ve komşularına ait yüzeysel (kelime şekli, ekler...) ve
isteğe bağlı olarak UPOS özellikleriyle temsil edilir.
"""
import io


def read_conll(path):
    """Chunk CoNLL dosyasını okur.
    Döndürür: list of sentences; her cümle list of (form, upos, chunk, clause)."""
    sents = []
    with io.open(path, encoding="utf-8") as f:
        cur = []
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if cur:
                    sents.append(cur)
                cur = []
                continue
            p = line.split("\t")
            if len(p) >= 5:
                cur.append((p[1], p[2], p[3], p[4]))   # form, upos, chunk, clause
        if cur:
            sents.append(cur)
    return sents


def word_shape(w):
    """Kelime şekli: harf->x/X, rakam->d, diğer aynı kalır."""
    out = []
    for ch in w:
        if ch.isdigit():
            out.append("d")
        elif ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        else:
            out.append(ch)
    return "".join(out)


def word2features(sent, i, use_upos=True, use_pred_pos=None):
    """sent: list of token-tuple; token[0]=form, token[1]=upos.
    use_pred_pos: verilirse UPOS yerine bu listedeki tahmini POS kullanılır."""
    w = sent[i][0]
    wl = w.lower()
    feats = {
        "bias": 1.0,
        "w.lower": wl,
        "w.istitle": w.istitle(),
        "w.isupper": w.isupper(),
        "w.isdigit": w.isdigit(),
        "w.shape": word_shape(w),
        "w.len": len(w),
        "suf1": wl[-1:],
        "suf2": wl[-2:],
        "suf3": wl[-3:],
        "suf4": wl[-4:],
        "pre1": wl[:1],
        "pre2": wl[:2],
        "pre3": wl[:3],
    }
    if use_upos:
        pos = use_pred_pos[i] if use_pred_pos is not None else sent[i][1]
        feats["upos"] = pos

    # önceki kelime
    if i > 0:
        pw = sent[i - 1][0]
        feats.update({
            "-1:w.lower": pw.lower(),
            "-1:w.istitle": pw.istitle(),
            "-1:suf3": pw.lower()[-3:],
        })
        if use_upos:
            ppos = use_pred_pos[i - 1] if use_pred_pos is not None else sent[i - 1][1]
            feats["-1:upos"] = ppos
    else:
        feats["BOS"] = True

    # iki önceki (sadece POS)
    if i > 1 and use_upos:
        p2 = use_pred_pos[i - 2] if use_pred_pos is not None else sent[i - 2][1]
        feats["-2:upos"] = p2

    # sonraki kelime
    if i < len(sent) - 1:
        nw = sent[i + 1][0]
        feats.update({
            "+1:w.lower": nw.lower(),
            "+1:w.istitle": nw.istitle(),
            "+1:suf3": nw.lower()[-3:],
        })
        if use_upos:
            npos = use_pred_pos[i + 1] if use_pred_pos is not None else sent[i + 1][1]
            feats["+1:upos"] = npos
    else:
        feats["EOS"] = True

    if i < len(sent) - 2 and use_upos:
        n2 = use_pred_pos[i + 2] if use_pred_pos is not None else sent[i + 2][1]
        feats["+2:upos"] = n2

    return feats


def sent2features(sent, use_upos=True, use_pred_pos=None):
    return [word2features(sent, i, use_upos, use_pred_pos) for i in range(len(sent))]


def sent2labels(sent, col):
    """col: 2=chunk, 3=clause, 1=upos"""
    return [tok[col] for tok in sent]


def sent2tokens(sent):
    return [tok[0] for tok in sent]
