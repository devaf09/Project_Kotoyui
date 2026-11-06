"""介入の研究 実験I25: 1 文を 1 枚に縮める — 文節のアレイを文法解析グレインが縮め、バッファ板（値 + ネスト）を作る。知らない語は仮の語板を育てる

案 2（文の圧縮と解凍）の 2 段目。算術（I19〜I23）と同じ形:
  - アレイ = 文節の並び（GiNZA の文節）。文節の種類 = 中心の語の品詞 / 最後の助詞（の・が・を…）か係り方（連体・連用）。
  - 命令は 1 つ「縮める（i の文節を右隣の i+1 に係らせて 1 枚にする）」。日本語は後ろに係るので、縮める順番が係り先を決める
    （「美しい ガラスの 靴」: 美しい を先に縮めると ガラス に係り、ガラスの を先に縮めると 美しい は 靴 に係る）。
  - 学び: GiNZA の係り受けを「正しい順番」として、前後 2 つの文節の種類の組で「縮める / 待つ」を覚える（I19 の段、器は I18 の組）。
    童話 8 話で学び、残り 4 話で試す（話を入れ替えて 3 回）。
  - バッファ板: 値 = 語板の明るさの和、参照 = 係った文節（ネスト）。z = 文の最後の 1 枚の値。
  - 仮の語板（持ち主の案、2026-11-06）: 語板に無い語は空の板を作り、読み進めるほど育てる。新しい名札は足さず、いまある名札だけを明るくする。
      形容詞 A に修飾される → 帯 B「性質:A」、A の「修飾先:クラス」を帯 D のクラスへ
      動詞 V に連体修飾される・V の が になる → 帯 C「V/する」、V の「主語:クラス」を D へ。V の を になる → 「V/される」、「目的語:クラス」を D へ
物差し: 試しの話での係り先の一致（文節単位、GiNZA と）、曖昧な例文 75 個で意図を選ぶ割合（I26 と比べる）、
        z だけで元の文を当てる（同じ話の文の中で）、仮の語板の帯 D（人か物か）。
結果: ../Project_Kotoyui_private/intervention/results/i25.jsonl
実行: python intervention/i25_sentence_board.py --tales <tales_v1_parsed.json> --labels <i24_ambig_labels.json> --boards <content_boards.json>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from i18_combo import ComboExp  # noqa: E402

FUNC_POS = {"ADP", "AUX", "SCONJ", "PART", "PUNCT", "CCONJ"}


def bunsetsu(nlp, text):
    """GiNZA の文節。返り値: 文節ごとの dict（中心の語、種類、係り先の文節番号）と、語 → 文節番号。"""
    import ginza
    doc = nlp(text)
    spans = list(ginza.bunsetu_spans(doc))
    tok2b = {}
    for b, sp in enumerate(spans):
        for t in sp:
            tok2b[t.i] = b
    out = []
    for b, sp in enumerate(spans):
        root = sp.root
        funcs = [t for t in sp if t.pos_ in FUNC_POS and t.pos_ != "PUNCT"]
        last = funcs[-1].lemma_ if funcs else {"acl": "連体", "advcl": "連用", "amod": "連体", "ROOT": "文末"}.get(root.dep_, root.dep_)
        hb = tok2b.get(root.head.i, b)
        content = [t for t in sp if t.pos_ not in FUNC_POS]
        out.append(dict(text=sp.text, root=root.lemma_, root_text=root.text, pos=root.pos_, kind=f"{root.pos_}/{last}",
                        head=hb if hb != b else -1, words=[(t.text, t.lemma_, t.pos_, t.dep_) for t in content],
                        compound="".join(t.text for t in content if t.pos_ in ("NOUN", "PROPN", "NUM")) or root.text))
    return out


def tree_ok(bs):
    """後ろに係り、交差しない木か（縮めるだけで作れる形）。根は最後の文節だけ。"""
    if sum(1 for b in bs if b["head"] == -1) != 1 or bs[-1]["head"] != -1:
        return False
    for i, b in enumerate(bs[:-1]):
        h = b["head"]
        if h <= i:
            return False
        for j in range(i + 1, h):
            if not (i < bs[j]["head"] <= h):
                return False
    return True


class Arr:
    def __init__(self, bs):
        self.bs = bs
        self.seq = list(range(len(bs)))                                # いまの並び（文節の番号 = その 1 枚の中心）
        self.head = {}                                                 # 縮めた文節 → 係った先

    def kind(self, p):
        if 0 <= p < len(self.seq):
            return self.bs[self.seq[p]]["kind"]
        return "端"

    def feats(self, p, k=2):
        return [f"{d}:{self.kind(p + d)}" for d in range(-k, k + 1)]

    def merge(self, p):
        self.head[self.seq[p]] = self.seq[p + 1]
        del self.seq[p]

    def oracle(self):
        """縮めてよい所: その文節の係り先が右隣で、その文節に係る文節がもう全部縮めてある。"""
        ok = []
        for p in range(len(self.seq) - 1):
            i = self.seq[p]
            if self.bs[i]["head"] == self.seq[p + 1] and all(j in self.head for j, b in enumerate(self.bs) if b["head"] == i):
                ok.append(p)
        return ok


class Decoder:
    def __init__(self):
        self.e = ComboExp("組")

    def learn(self, bs):
        a = Arr(bs)
        while len(a.seq) > 1:
            ok = a.oracle()
            if not ok:
                return False
            for p in range(len(a.seq) - 1):
                self.e.add(a.feats(p), "命令", "縮める" if p in ok else "待つ")
            a.merge(ok[0])
        return True

    def score(self, a, p):
        s = self.e.scores(a.feats(p), "命令")
        return s.get("縮める", -50.0) - s.get("待つ", -50.0) if s else 0.0

    def parse(self, bs):
        a = Arr(bs)
        while len(a.seq) > 1:
            sc = [self.score(a, p) for p in range(len(a.seq) - 1)]
            a.merge(int(np.argmax(sc)))
        return a.head


def nearest(bs):
    return {i: i + 1 for i in range(len(bs) - 1)}


# ---- 板 ----

class Store:
    def __init__(self, path, slots):
        self.b = {x["name"]: x["attrs"] for x in json.load(open(path, encoding="utf-8"))}
        self.slots = {band: set(v) for band, v in slots.items()}
        self.temp = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))   # 話ごと・語ごと・帯ごとの仮の語板（数え）
        self.how = Counter()

    def lookup(self, word, lemma):
        for w in (word, lemma):
            if w in self.b:
                return self.b[w]
        return None

    def board(self, tale, b):
        """文節の中心の語の板。無ければ仮の語板（割合にして返す）。"""
        bd = self.lookup(b["compound"], b["root"])
        if bd is not None:
            self.how["語板"] += 1
            return bd
        self.how["仮の語板"] += 1
        t = self.temp[tale][b["compound"]]
        out = {}
        for band, c in t.items():
            n = sum(c.values())
            out[band] = {k: v / n for k, v in c.items()} if n else {}
        return out

    def grow(self, tale, dep_b, head_b):
        """仮の語板を育てる: dep_b が head_b に係ったとき、どちらかが語板に無ければ、もう一方から名札を明るくする。"""
        dep_bd = self.lookup(dep_b["compound"], dep_b["root"])
        head_bd = self.lookup(head_b["compound"], head_b["root"])
        if head_bd is None and head_b["pos"] in ("NOUN", "PROPN") and dep_bd is not None:   # 修飾される名詞
            t = self.temp[tale][head_b["compound"]]
            if dep_b["pos"] == "ADJ" or dep_b["kind"].endswith("/連体") and dep_b["pos"] == "ADJ":
                self._add(t, "B", "性質:" + dep_b["root"])
                for k, v in dep_bd.get("C", {}).items():
                    if k.startswith("修飾先:"):
                        self._add(t, "D", "クラス:" + k.split(":", 1)[1], v)
            elif dep_b["pos"] == "VERB":
                self._add(t, "C", dep_b["root"] + "/する")
                for k, v in dep_bd.get("C", {}).items():
                    if k.startswith("主語:"):
                        self._add(t, "D", "クラス:" + k.split(":", 1)[1], v)
        if dep_bd is None and dep_b["pos"] in ("NOUN", "PROPN") and head_bd is not None and head_b["pos"] == "VERB":
            t = self.temp[tale][dep_b["compound"]]
            case = dep_b["kind"].split("/")[-1]
            if case in ("が", "は"):
                self._add(t, "C", head_b["root"] + "/する")
                pre = "主語:"
            elif case == "を":
                self._add(t, "C", head_b["root"] + "/される")
                pre = "目的語:"
            else:
                return
            for k, v in head_bd.get("C", {}).items():
                if k.startswith(pre):
                    self._add(t, "D", "クラス:" + k.split(":", 1)[1], v)

    def _add(self, t, band, key, v=1.0):
        if key in self.slots.get(band, ()):                           # 新しい名札は足さない
            t[band][key] += v


def vec(bd, index):
    x = np.zeros(len(index))
    for band, d in bd.items():
        for k, v in d.items():
            j = index.get((band, k))
            if j is not None:
                x[j] += v
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tales", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--boards", required=True)
    ap.add_argument("--slots", default=None)
    ap.add_argument("--out", default="../Project_Kotoyui_private/intervention/results/i25.jsonl")
    a = ap.parse_args()
    import spacy
    nlp = spacy.load("ja_ginza")
    cache_path = os.path.join(os.path.dirname(a.tales), "i25_bunsetsu_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}

    def bunsetsu_c(text):
        if text not in cache:
            cache[text] = bunsetsu(nlp, text)
        return cache[text]
    tales = json.load(open(a.tales, encoding="utf-8"))
    labels = json.load(open(a.labels, encoding="utf-8"))
    slots = json.load(open(a.slots or os.path.join(os.path.dirname(a.boards), "slots.json"), encoding="utf-8"))["slots"]
    sents = []                                                          # (話, 段, 文節)
    skipped = 0
    for ti, t in enumerate(tales):
        for lv, ss in t["versions"].items():
            for s in ss:
                bs = bunsetsu_c(s)
                if len(bs) < 2 or not tree_ok(bs):
                    skipped += 1
                    continue
                sents.append((ti, lv, bs))
    rec = dict(exp="I25", n_sents=len(sents), skipped=skipped, folds=[])
    # 1. 係り先: 8 話で学び 4 話で試す（3 通り）
    order = list(range(len(tales)))
    for fold in range(3):
        test_t = set(order[fold * 4:(fold + 1) * 4])
        dec = Decoder()
        for ti, lv, bs in sents:
            if ti not in test_t:
                dec.learn(bs)
        tot = Counter()
        for ti, lv, bs in sents:
            if ti not in test_t:
                continue
            gold = {i: b["head"] for i, b in enumerate(bs) if b["head"] != -1}
            for name, h in (("グレイン", dec.parse(bs)), ("近い方", nearest(bs))):
                tot[name] += sum(1 for i, g in gold.items() if h.get(i) == g)
                tot[name + "_文"] += all(h.get(i) == g for i, g in gold.items())
            tot["n"] += len(gold)
            tot["n_文"] += 1
        rec["folds"].append(dict(test=[tales[i]["name"] for i in sorted(test_t)], uas=tot["グレイン"] / tot["n"], uas_near=tot["近い方"] / tot["n"],
                                 sent=tot["グレイン_文"] / tot["n_文"], sent_near=tot["近い方_文"] / tot["n_文"]))
    # 2. 曖昧な例文: 全部の話で学んだグレインで読む
    dec = Decoder()
    for ti, lv, bs in sents:
        dec.learn(bs)
    amb = Counter()
    for ti, t in enumerate(tales):
        for k, it in enumerate(t["ambig"]):
            lab = labels.get(f"{ti}.{k}")
            if not lab or lab[0] != "attach":
                continue
            bs = bunsetsu_c(it["sent"])
            if len(bs) < 2:
                continue

            def find(w):
                return [i for i, b in enumerate(bs) if any(x[0] == w for x in b["words"]) or w in b["text"]]
            m, ia, ib = find(lab[1]), find(lab[2]), find(lab[3])
            if not (m and ia and ib):
                amb["文節が取れない"] += 1
                continue
            m = m[0]
            ia = [i for i in ia if i != m]
            ib = [i for i in ib if i != m]
            if not ia or not ib or ia[0] == ib[0]:
                amb["同じ文節"] += 1
                continue
            h = dec.parse(bs)
            x = m
            got = "どちらでもない"
            for _ in range(len(bs)):
                x = h.get(x)
                if x is None:
                    break
                if x in (ia[0], ib[0]):
                    got = "意図" if x == ia[0] else "誤読"
                    break
            amb[got] += 1
    rec["ambig"] = dict(amb)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    # 3. バッファ板と z: 文を 1 枚に縮め、z だけで同じ話の文の中から元の文を当てる。仮の語板を育てる
    st = Store(a.boards, slots)
    index = {}
    for band, ks in slots.items():
        for k in ks:
            index[(band, k)] = len(index)
    for ti, lv, bs in sents:                                           # 読み進めて仮の語板を育てる（GiNZA の係り先で）
        for i, b in enumerate(bs):
            if b["head"] != -1:
                st.grow(ti, b, bs[b["head"]])
    zhit = Counter()
    for ti in range(len(tales)):
        group = [(lv, bs) for t2, lv, bs in sents if t2 == ti and lv == "20"]
        Z = []
        for lv, bs in group:
            z = sum(vec(st.board(ti, b), index) for b in bs)
            Z.append(z / (np.linalg.norm(z) + 1e-9))
        Z = np.array(Z)
        rng = np.random.default_rng(ti)
        for j in range(len(Z)):
            sims = Z @ Z[j]
            zhit["自分"] += int(np.argmax(sims) == j)
            noisy = Z[j] + rng.normal(0, 0.05, Z.shape[1])                   # 雑音を足した z
            zhit["雑音 0.05"] += int(np.argmax(Z @ noisy) == j)
            zhit["n"] += 1
    rec["z_find"] = {k: v / zhit["n"] for k, v in zhit.items() if k != "n"}
    rec["z_find_n"] = zhit["n"]
    rec["board_lookup"] = dict(st.how)
    temp = {}
    for ti, words in st.temp.items():
        for w, bands in words.items():
            d = bands.get("D", Counter())
            n = sum(sum(c.values()) for c in bands.values())
            temp[f"{tales[ti]['name']}:{w}"] = dict(n=round(n, 2), D=[k for k, _ in d.most_common(2)],
                                                    top=[k for band in ("B", "C") for k, _ in bands.get(band, Counter()).most_common(2)])
    rec["temp_boards"] = temp
    print(json.dumps({k: v for k, v in rec.items() if k != "temp_boards"}, ensure_ascii=False, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
