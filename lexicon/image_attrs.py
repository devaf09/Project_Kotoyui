"""語板・機能板・品詞板に「像の偏り」（話し手の像がよく使う語か）の属性を足す（2026-11-08）

依頼: 介入の研究 requests/word_board_image_attributes.md（材料は I27w の語 × 像 の表、_wn 由来なので出す物はすべて非公開）。
持ち主（2026-11-08）: 「語板、機能板、品詞板にそれぞれの像が優先的に選ぶ属性か … 像がよく使う属性を追加するのはどう思う？」「始めてOK」

材料の表（i27w_word_table_wn.tsv）: 列 pos, lemma, count, stable（半分どうしの向きの合い方）, 像1〜像8 の偏り
  （偏り b_k = log(その語での像 k の割合 / 全部の語での像 k の割合 q_k)）。
足し方:
  語での像の割合 s_k = q_k · exp(b_k) を、出口の明るさにそのまま使える割合に戻す（q_k は --prior、無ければ同じ割合）。
  帯 S（文体の像）: 像 6 を除いた 7 つの s_k を割合にしたもの。
  帯 H（息づかい）: {息づかい: s_6, ふつう: 1 − s_6}。像 6 は人の型ではないので帯を分ける（依頼の 4）。
  帯 H を種の平均で作る版（--breath_tables / --breath_priors、例 K=16 の種 0〜7）: 種ごとに息づかいの像を目印の語尾で見つけ、
         語ごとの息づかいの割合を種で平均する。出なかった種は 0 として数える（種を選ぶ手を入れない）。帯 S は主の表の像をすべて使う。
  足切り: stable ≥ 0.5 かつ count ≥ 20。名詞はさらに「出る作品の数」≥ --min_works（作品の話題を文体と取り違えないため）。
         作品の数の表（--works）が無ければ、名詞には足さない。
  名前（尊敬語・謙譲語・現代語・古風…）は手で付けない。像の番号だけで持つ（依頼の 5）。
  機能板: 見出し（語・種類）の合う行の割合を、回数の重みで足して割合に戻す（光の和の融合と同じ）。働き（func）ごとの束も出す。
  品詞板: 品詞ごとに、足切りを通った語の割合を回数の重みで足して割合に戻す。
確かめ（--half_a / --half_b、作品を半分ずつにして作った同じ形の表）: 足した値（s_k）の、半分どうしの相関を像ごとに出す。

出す物（--out、名前は _wn で終わること）: content_boards.json（帯 S・H を足した語板）、slots.json、func_boards.json、pos_boards.json、
  image_groups.json（働きごとの束）、image_attrs_report.json、sources.json。
実行: python lexicon/image_attrs.py --table <i27w_word_table_wn.tsv> --boards <data/boards/v2_3b> --special <data/boards>
        --works <語ごとの作品の数.tsv> --out <data/boards/v2_3b_img_wn> [--prior q1,...,q8] [--half_a a.tsv --half_b b.tsv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np

K = 8
BREATH = 6                                 # 息づかいの像の番号。main で表から見つけ直す（像の番号は k 平均の種で入れ替わるため）
STYLE = [k for k in range(1, K + 1) if k != BREATH]
BREATH_MARKS = ("っ", "いっ", "すっ", "んっ", "ょ")                # 息づかいの像を見つける目印（I27w のらしい語尾）
H_POS_SPECIAL = {"助動詞", "助詞", "感動詞", "副詞"}               # 品詞板で帯 H を付ける品詞（語尾の属性に寄せる）
FUNC_UPOS = ("AUX", "PART", "ADP", "SCONJ", "INTJ")


def find_breath(rows, marks, min_count=100):
    """目印の語での偏りの平均がいちばん大きい像を、息づかいの像とする。"""
    acc = defaultdict(list)
    for r in rows:
        if r["lemma"] in marks and r["count"] >= min_count:
            for k in r["b"]:
                acc[k].append(r["b"][k])
    if not acc:
        return None, {}
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    return max(mean, key=mean.get), {f"像{k}": round(v, 3) for k, v in mean.items()}


def image_marks(rows, n=6, min_count=200):
    """像ごとの目印（機能語のうち偏りの大きい語）。像の番号が変わっても、これで照合する。"""
    fr = [r for r in rows if r["count"] >= min_count and (r["pos"] in FUNC_UPOS or r["pos"].startswith(("助詞", "助動詞")))]
    return {f"像{k}": [r["lemma"] for r in sorted(fr, key=lambda r: -r["b"][k])[:n]] for k in range(1, K + 1)}
IMG_COL = re.compile(r"^(?:像|img|image|b)_?(\d+)")


def hira(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def read_table(path, k_expect=K):
    """表を読む。列の名前の揺れ（像1 / 像1_偏り / img1 …）を吸収する。行 = dict(pos, lemma, count, stable, b={k: 偏り})。"""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)   # 語が " のときに行がつながらないように
        head = next(r)
        col = {h.strip(): i for i, h in enumerate(head)}
        img = {}
        for h, i in col.items():
            m = IMG_COL.match(h)
            if m and int(m.group(1)) >= 1 and int(m.group(1)) not in img:
                img[int(m.group(1))] = i
        need = ("pos", "lemma", "count")
        if any(n not in col for n in need) or sorted(img) != list(range(1, len(img) + 1)) or (k_expect and len(img) != k_expect):
            raise SystemExit(f"表の列が読めない: {head}")
        for x in r:
            if len(x) < len(head):
                continue
            try:
                rows.append(dict(pos=x[col["pos"]], lemma=x[col["lemma"]], count=float(x[col["count"]]),
                                 stable=float(x[col["stable"]]) if "stable" in col and x[col["stable"]] not in ("", "nan") else float("nan"),
                                 b={k: float(x[i]) for k, i in img.items()}))
            except ValueError:
                continue
    return rows


def read_works(path):
    """語ごとの作品の数。列 lemma, n_works（pos があれば (pos, lemma) でも引く）。"""
    if not path:
        return None
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            n = int(float(r["n_works"]))
            out[(r.get("pos", ""), r["lemma"])] = n
            out[("", r["lemma"])] = max(out.get(("", r["lemma"]), 0), n)
    return out


def is_noun(pos):
    return pos.startswith(("名詞", "NOUN", "PROPN")) or pos in ("名詞",)


def shares(b, prior):
    s = np.array([prior[k - 1] * math.exp(b[k]) for k in range(1, K + 1)])
    return s / s.sum()                    # 表の偏りが丸めてあっても和が 1 になるように


def bands_of(s, h=None):
    if len(s) == K + 1:                                              # 末尾 = 種の平均の息づかいの割合（--breath_tables のとき）
        s, h = s[:K], s[K]
    st = np.array([s[k - 1] for k in STYLE])
    st = st / st.sum()
    S = {f"像{k}": round(float(v), 4) for k, v in sorted(zip(STYLE, st), key=lambda kv: -kv[1])}
    if h is not None:                                                # 帯 H を種の平均から作る版
        return S, {"息づかい": round(float(h), 4), "ふつう": round(float(1 - h), 4)}
    if BREATH is None:                                               # 息づかいの像が無い種では、帯 H を作らない
        return S, None
    H = {"息づかい": round(float(s[BREATH - 1]), 4), "ふつう": round(float(1 - s[BREATH - 1]), 4)}
    return S, H


def load_prior(x, k):
    """像の割合 q: , 区切りの文字列か、JSON（数の並び、または {"q": [...]} / {"prior": [...]}）のファイル。"""
    if os.path.exists(x):
        v = json.load(open(x, encoding="utf-8"))
        if isinstance(v, dict):
            v = v.get("q") or v.get("prior") or list(v.values())
    else:
        v = [float(t) for t in x.split(",")]
    v = [float(t) for t in v]
    assert len(v) == k, f"像の割合の数が {len(v)}（{k} のはず）: {x}"
    return v


def breath_by_seeds(tables, priors, marks, bmin):
    """K の大きい表（種ごと）で息づかいの像を目印の語尾で見つけ、語ごとの息づかいの割合を種で平均する。
    息づかいの像が出なかった種は 0 として数える（種を選ぶ手を入れないため）。表に無い語は、その種を数えない。"""
    acc = defaultdict(list)
    seeds = []
    for path, pr in zip(tables, priors):
        rows = read_table(path, k_expect=None)
        k = len(rows[0]["b"])
        q = load_prior(pr, k)
        found, score = find_breath(rows, marks)
        ok = found is not None and score[f"像{found}"] >= bmin
        seeds.append(dict(表=os.path.basename(path), K=k, 息づかいの像=(f"像{found}" if ok else "なし"), 目印の偏り=score))
        for r in rows:
            if ok:
                e = np.array([q[j - 1] * math.exp(r["b"][j]) for j in range(1, k + 1)])
                acc[(r["pos"], r["lemma"])].append(float(e[found - 1] / e.sum()))
            else:
                acc[(r["pos"], r["lemma"])].append(0.0)
    return {key: float(np.mean(v)) for key, v in acc.items()}, seeds


def is_frag(lemma):
    """帯 H から外す切れ端: 原形が っ/ッ で終わる 2 文字以上の語（I27w の像を通さない属性と同じ決まり。手の一覧は使わない）。"""
    return len(lemma) >= 2 and lemma[-1] in "っッ"


def h_only(items):
    """帯 H だけを、切れ端を除いた (割合, 回数, 語) から作る。帯 S は全部の語から作る。"""
    it = [(x, w) for x, w, lem in items if not is_frag(lem)]
    return bands_of(fuse(it))[1] if it else None


def put(attrs, ks, kh, S, H):
    attrs[ks] = S
    if H is not None:
        attrs[kh] = H


def passes(r, works, a):
    if not (r["count"] >= a.min_count and r["stable"] >= a.min_stable):
        return False, "回数・向き"
    if is_noun(r["pos"]):
        if works is None:
            return False, "名詞（作品の数が無い）"
        n = works.get((r["pos"], r["lemma"]), works.get(("", r["lemma"]), 0))
        if n < a.min_works:
            return False, "名詞（作品が少ない）"
    return True, "通る"


# ---- 機能板の見出しと表の品詞を合わせる ----
KIND_POS = {   # 機能板の種類 → 表の品詞（UniDic の書き方・GiNZA の UPOS のどちらでも）
    "補助動詞": ("動詞", "VERB", "AUX"), "補助形容詞": ("形容詞", "ADJ", "AUX"), "助動詞": ("助動詞", "AUX", "形状詞-助動詞語幹", "名詞-助動詞語幹"),
    "接続詞": ("接続詞", "CCONJ", "SCONJ"), "フィラー": ("感動詞", "INTJ"), "形式名詞": ("名詞", "NOUN"),
    "指示詞": ("代名詞", "連体詞", "副詞", "形状詞", "PRON", "DET", "ADV", "ADJ"),
}
PART_UPOS = {"格助詞": ("ADP",), "係助詞": ("ADP", "PART"), "副助詞": ("ADP", "PART"), "接続助詞": ("SCONJ", "CCONJ"),
             "終助詞": ("PART",), "準体助詞": ("SCONJ", "PART", "ADP")}


def pos_fits(kind, pos):
    if kind in PART_UPOS:                                            # 助詞は「助詞-終助詞」のように細かい品詞があれば、それで合わせる
        if pos.startswith("助詞"):
            return kind in pos or pos == "助詞"
        return pos in PART_UPOS[kind]
    return any(pos == p or pos.startswith(p + "-") or pos.startswith(p) for p in KIND_POS.get(kind, ()))


def fuse(items):
    """(割合, 回数) を回数の重みで足し、割合に戻す（光の和）。"""
    if not items:
        return None
    t = sum(w for _, w in items)
    return sum(s * w for s, w in items) / t


def pos_key(pos):
    """表の品詞 → 品詞板の名前（UniDic の 1〜2 段目）。UPOS なら 1 段目だけ。"""
    p = pos.split("-")
    UP = {"NOUN": "名詞", "PROPN": "名詞", "VERB": "動詞", "ADJ": "形容詞", "ADV": "副詞", "AUX": "助動詞", "ADP": "助詞", "PART": "助詞",
          "SCONJ": "助詞", "CCONJ": "接続詞", "INTJ": "感動詞", "PRON": "代名詞", "DET": "連体詞", "NUM": "名詞"}
    if p[0] in UP:
        return UP[p[0]], False
    two = p[0] + ("-" + p[1] if len(p) > 1 and p[0] in ("名詞", "助詞", "接尾辞", "補助記号", "動詞", "形容詞", "形状詞") else "")
    return two, True


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--boards", required=True, help="語板の置き場（content_boards.json・slots.json がある所）")
    ap.add_argument("--special", default=None, help="func_boards.json・pos_boards.json の置き場")
    ap.add_argument("--works", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior", default=None, help="全部の語での像の割合 q1..q8（, 区切り）。無ければ同じ割合")
    ap.add_argument("--min_count", type=float, default=20)
    ap.add_argument("--min_stable", type=float, default=0.5)
    ap.add_argument("--min_works", type=int, default=20)
    ap.add_argument("--breath", type=int, default=None, help="息づかいの像の番号（無ければ目印の語で見つける。0 = 息づかいの像なし）")
    ap.add_argument("--breath_min", type=float, default=1.0, help="目印の語の偏りの平均がこれ未満なら、息づかいの像は無いとみなす")
    ap.add_argument("--breath_marks", default=",".join(BREATH_MARKS))
    ap.add_argument("--breath_tables", nargs="*", default=None, help="帯 H を種の平均で作るための表（例 K=16 の種 0〜7）")
    ap.add_argument("--breath_priors", nargs="*", default=None, help="その表ごとの像の割合（, 区切りか JSON のファイル）")
    ap.add_argument("--h_pos", default="副詞,動詞,形容詞", help="語板で帯 H を付ける品詞（名詞には付けない）。上位に話題の語が残るなら 副詞 だけにする")
    ap.add_argument("--half_a", default=None)
    ap.add_argument("--half_b", default=None)
    a = ap.parse_args()
    if not os.path.basename(os.path.normpath(a.out)).endswith("_wn"):
        raise SystemExit("像の偏りは _wn の資料から作るので、置き場の名前は _wn で終わること")
    os.makedirs(a.out, exist_ok=True)
    prior = [float(x) for x in a.prior.split(",")] if a.prior else [1.0 / K] * K
    assert len(prior) == K
    rows = read_table(a.table)
    global BREATH, STYLE
    found, breath_score = find_breath(rows, set(a.breath_marks.split(",")))
    if a.breath == 0 or (a.breath is None and (found is None or breath_score[f"像{found}"] < a.breath_min)):
        BREATH = None                                                # 息づかいの像が無い: 8 つの像をすべて帯 S に入れ、帯 H は作らない
        print(f"息づかいの像なし（目印の偏り {breath_score}）。帯 H は作らない")
    else:
        BREATH = a.breath or found
    STYLE = [k for k in range(1, K + 1) if k != BREATH]
    marks = image_marks(rows)
    hmap, seeds = None, None
    if a.breath_tables:
        if not a.breath_priors or len(a.breath_priors) != len(a.breath_tables):
            raise SystemExit("--breath_tables と同じ数の --breath_priors が要る")
        hmap, seeds = breath_by_seeds(a.breath_tables, a.breath_priors, set(a.breath_marks.split(",")), a.breath_min)
        BREATH = None                                                # 帯 S は主の表の像をすべて使い、帯 H は種の平均から作る
        STYLE = list(range(1, K + 1))
    works = read_works(a.works)
    why = Counter()
    ok = []
    for r in rows:
        p, w = passes(r, works, a)
        why[w] += 1
        if p:
            r["s"] = shares(r["b"], prior)
            if hmap is not None:
                hv = hmap.get((r["pos"], r["lemma"]))
                why["帯 H の表に無い語（息づかい 0）"] += hv is None
                r["s"] = np.append(r["s"], hv if hv is not None else 0.0)
            ok.append(r)
    by_lemma = defaultdict(list)
    for r in ok:
        by_lemma[r["lemma"]].append(r)

    # ---- 語板 ----
    boards = json.load(open(os.path.join(a.boards, "content_boards.json"), encoding="utf-8"))
    POS_UP = {"名詞": ("名詞", "NOUN", "PROPN"), "動詞": ("動詞", "VERB"), "形容詞": ("形容詞", "ADJ", "形状詞"), "副詞": ("副詞", "ADV")}
    n_add = Counter()
    h_pos = set(a.h_pos.split(","))
    n_h = 0
    for b in boards:
        cand = by_lemma.get(b["name"], [])
        fit = [r for r in cand if any(r["pos"].startswith(p) for p in POS_UP.get(b["pos"], (b["pos"],)))] or \
              ([] if b["pos"] == "名詞" else cand)                     # 名詞は品詞の合う行だけ（足切りを通った名詞の行だけを使う）
        fit = [r for r in fit if "非自立" not in r["pos"]] or fit          # 語板は本動詞の行を使う（補助動詞は機能板）
        if not fit:
            continue
        s = fuse([(r["s"], r["count"]) for r in fit])
        S, H = bands_of(s)
        if H is not None and (b["pos"] not in h_pos or is_frag(b["name"])):
            H = None                                                 # 帯 H は名詞に付けない（話題を拾うため）。切れ端も外す
        n_h += H is not None
        put(b["attrs"], "S", "H", S, H)
        b.setdefault("n_obs", {})["S"] = round(sum(r["count"] for r in fit), 1)
        n_add[b["pos"]] += 1
    json.dump(boards, open(os.path.join(a.out, "content_boards.json"), "w", encoding="utf-8"), ensure_ascii=False)
    slots = json.load(open(os.path.join(a.boards, "slots.json"), encoding="utf-8"))
    slots["slots"]["S"] = [f"像{k}" for k in STYLE]
    if BREATH is not None or hmap is not None:
        slots["slots"]["H"] = ["息づかい", "ふつう"]
    slots["image_marks"] = marks                                    # 像の番号は種で入れ替わるので、目印の語で照合する
    json.dump(slots, open(os.path.join(a.out, "slots.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_slots = sum(len(v) for v in slots["slots"].values())

    # ---- 機能板・品詞板 ----
    rep_special = {}
    if a.special:
        F = json.load(open(os.path.join(a.special, "func_boards.json"), encoding="utf-8"))
        groups = defaultdict(list)
        n_f = 0
        for fb in F:
            word, kind = fb["word"], fb["kind"]
            keys = {word, hira(word)}
            fit = [r for k in keys for r in by_lemma.get(k, []) if pos_fits(kind, r["pos"])]
            if kind in ("補助動詞", "補助形容詞", "形式名詞"):                    # 非自立の行があれば、それだけを使う（くださる の本動詞と分ける）
                fit = [r for r in fit if "非自立" in r["pos"]] or fit
            if not fit:
                continue
            s = fuse([(r["s"], r["count"]) for r in fit])
            S, H = bands_of(s)
            if is_frag(word):
                H = None
            put(fb["attrs"], "像", "息づかい", S, H)
            fb["n_image"] = round(sum(r["count"] for r in fit), 1)
            groups[fb["func"]].append((s, fb["n_image"], word))
            n_f += 1
        G = []
        for g, it in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            S = bands_of(fuse([(x, w) for x, w, _ in it]))[0]
            H = h_only(it)
            G.append(dict(name=g, kind="働きの束", n_boards=len(it), n_image=round(sum(w for _, w, _ in it), 1), attrs={"像": S, **({"息づかい": H} if H else {})}))
        json.dump(F, open(os.path.join(a.out, "func_boards.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(G, open(os.path.join(a.out, "image_groups.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        P = json.load(open(os.path.join(a.special, "pos_boards.json"), encoding="utf-8"))
        acc2, acc1 = defaultdict(list), defaultdict(list)
        for r in ok:
            k, fine = pos_key(r["pos"])
            (acc2 if fine else acc1)[k].append((r["s"], r["count"], r["lemma"]))
            if fine:
                acc1[k.split("-")[0]].append((r["s"], r["count"], r["lemma"]))
        n_p = 0
        for pb in P:
            it = acc2.get(pb["name"]) or acc1.get(pb["name"].split("-")[0])
            if not it:
                continue
            S = bands_of(fuse([(x, w) for x, w, _ in it]))[0]
            H = h_only(it) if pb["name"].split("-")[0] in H_POS_SPECIAL | h_pos else None
            put(pb["attrs"], "像", "息づかい", S, H)
            pb["n_image_words"] = len(it)
            pb["image_unit"] = "品詞の 2 段目" if pb["name"] in acc2 else "品詞の 1 段目"
            n_p += 1
        json.dump(P, open(os.path.join(a.out, "pos_boards.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        rep_special = dict(機能板=f"{n_f} / {len(F)}", 働きの束=len(G), 品詞板=f"{n_p} / {len(P)}")

    # ---- 確かめ: 半分どうしの相関 ----
    half = None
    if a.half_a and a.half_b:
        ha = {(r["pos"], r["lemma"]): r for r in read_table(a.half_a)}
        hb = {(r["pos"], r["lemma"]): r for r in read_table(a.half_b)}
        keep = {(r["pos"], r["lemma"]) for r in ok}
        half = {}
        for name, sel in (("足切りを通った語", lambda k: k in keep), ("全部の語（両方で 20 回以上）",
                          lambda k: ha[k]["count"] >= a.min_count and hb[k]["count"] >= a.min_count)):
            ks = [k for k in ha if k in hb and sel(k)]
            SA = np.array([shares(ha[k]["b"], prior) for k in ks]) if ks else np.zeros((0, K))
            SB = np.array([shares(hb[k]["b"], prior) for k in ks]) if ks else np.zeros((0, K))
            per = {f"像{k}": corr(np.log(SA[:, k - 1] + 1e-9), np.log(SB[:, k - 1] + 1e-9)) for k in range(1, K + 1)} if ks else {}
            half[name] = dict(語=len(ks), 像ごと=per, 汎用板の候補=[k for k, v in per.items() if v is not None and v >= 0.5])

    rep = dict(息づかいの像=(f"像{BREATH}" if BREATH else ("種の平均（帯 H）" if hmap is not None else "なし")), 息づかいの種=seeds, 息づかいの目印の偏り=breath_score, 像の目印=marks, 表の行=len(rows), 足切り=dict(why), 語板に足した語=dict(n_add), 帯Hを付けた語板=n_h, 名札=n_slots, 出口の上限=2040, **rep_special,
               像の割合の前提=("--prior" if a.prior else "同じ割合（全部の語での割合が無いため）"), 半分どうし=half,
               足切りの値=dict(回数=a.min_count, 向き=a.min_stable, 名詞の作品の数=a.min_works))
    json.dump(rep, open(os.path.join(a.out, "image_attrs_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(dict(公開してよいか=False, 公開しない資料=True, 資料=dict(語板=a.boards, 像の表=a.table, 作品の数=a.works),
                   名前の決まり="公開しない資料を混ぜた語板は、置き場の名前が _wn で終わる"),
              open(os.path.join(a.out, "sources.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if n_slots > 2040:
        print("注意: 名札が出口の上限を越えた")


if __name__ == "__main__":
    main()
