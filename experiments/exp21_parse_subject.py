"""実験21（非公開の材料）: 文法解析（係り受け）の手がかりを足して、主語を隠した発言の人称を当てる（持ち主の計画 (3)、2026-11-04）

実験19・20 と同じ問（同じ seed で同じ問）。主語を消した文を GiNZA（MIT）で係り受け解析し、
文の中心の述語（ROOT）とそれに付く助動詞・補助動詞・終わりの形、ほかの主語（nsubj）の有無、述語の品詞を「構造の手がかり」にする。
  例: ROOT=嬉しい（形容詞）・AUX=です → 話し手らしい / ROOT=行く・AUX=たい → 話し手 / ROOT=言う・AUX=らしい → 第三者
段:
  --stage dump  : 問を作って ../data/persona/subject_items_s<seed>.jsonl に書く（いつもの環境）
  --stage parse : GiNZA の環境（venv_ginza の python）で解析し、../data/persona/subject_feats_s<seed>.jsonl に書く
  --stage eval  : 数えた表・光の 2DWIG に、構造の手がかりの袋を足して測る（いつもの環境、--device）
結果: ../results/exp21.txt
"""
import argparse
import json
import os
import random
import re
import sys
import time

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
DATA = os.path.join(PRIV, "data", "persona")


def dump(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp19_hidden_subject import PAIRS, mask
    rng = random.Random(a.seed)
    seen = set()
    tr, te = {k: [] for k in range(3)}, {k: [] for k in range(3)}
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        for t in (r["prev"], r["reply"]):
            if t in seen:
                continue
            seen.add(t)
            mt = mask(t)
            if mt:
                (tr if r["split"] == "train" else te)[mt[1]].append((mt[0], mt[1], t, r["src"]))
    n_tr = min(a.max_per_class, min(len(v) for v in tr.values()))
    n_te = min(5000, min(len(v) for v in te.values()))
    trs, tes = [], []
    for k in range(3):                                   # 実験19 と同じ乱数の使い方（同じ問になる）
        rng.shuffle(tr[k])
        rng.shuffle(te[k])
        trs += tr[k][:n_tr]
        tes += te[k][:n_te]
    rng.shuffle(trs)
    with open(os.path.join(DATA, f"subject_items_s{a.seed}.jsonl"), "w", encoding="utf-8") as f:
        for split, items in (("train", trs), ("test", tes)):
            for x in items:
                f.write(json.dumps(dict(split=split, text=x[0], label=x[1]), ensure_ascii=False) + "\n")
    print("書いた", len(trs), len(tes))


def parse(a):
    import spacy
    nlp = spacy.load("ja_ginza")
    items = [json.loads(l) for l in open(os.path.join(DATA, f"subject_items_s{a.seed}.jsonl"), encoding="utf-8")]
    t0 = time.perf_counter()
    with open(os.path.join(DATA, f"subject_feats_s{a.seed}.jsonl"), "w", encoding="utf-8") as f:
        for i, doc in enumerate(nlp.pipe([x["text"] for x in items], batch_size=256)):
            feats = []
            for sent in doc.sents:
                root = sent.root
                feats += [f"R:{root.lemma_}", f"RP:{root.pos_}"]
                for c in root.children:
                    if c.dep_ in ("aux", "cop", "mark"):
                        feats.append(f"A:{c.lemma_}")
                    if c.dep_ in ("nsubj", "csubj"):
                        feats.append(f"S:{c.pos_}")
                    if c.dep_ in ("obj", "iobj", "obl"):
                        feats.append(f"O:{c.dep_}")
                last = [t for t in sent if t.pos_ not in ("PUNCT", "SYM")]
                if last:
                    feats.append(f"E:{last[-1].lemma_}")
                infl = root.morph.get("Inflection")
                if infl:
                    feats.append(f"I:{infl[0]}")
            f.write(json.dumps(feats, ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print(i + 1, f"{time.perf_counter() - t0:.0f} 秒", flush=True)


def evaluate(a):
    import torch
    KOTO = KOTO_ROOT
    sys.path.insert(0, os.path.join(KOTO, "experiments"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import zlib
    from exp01_wig_bag import L, bag, p2, to_field
    from exp19_hidden_subject import PERSONS, rule_scores
    items = [json.loads(l) for l in open(os.path.join(DATA, f"subject_items_s{a.seed}.jsonl"), encoding="utf-8")]
    feats = [json.loads(l) for l in open(os.path.join(DATA, f"subject_feats_s{a.seed}.jsonl"), encoding="utf-8")]
    tr = [i for i, x in enumerate(items) if x["split"] == "train"]
    te = [i for i, x in enumerate(items) if x["split"] == "test"]
    y = torch.tensor([items[i]["label"] for i in tr])
    yt = torch.tensor([items[i]["label"] for i in te])

    def fbag(ids):
        X = torch.zeros(len(ids), L * L, dtype=torch.float16)
        for r, i in enumerate(ids):
            for f in feats[i]:
                X[r, zlib.crc32(("〒" + f).encode("utf-8")) % (L * L)] += 1
        return X
    runs = []

    def report(name, pred, extra=None):
        pred = torch.as_tensor(pred)
        per = {PERSONS[k]: 100.0 * (pred[yt == k] == k).float().mean().item() for k in range(3)}
        r = dict(method=name, acc=100.0 * (pred == yt).float().mean().item(), per_class=per, **(extra or {}))
        runs.append(r)
        print(name, round(r["acc"], 1), {k: round(v, 1) for k, v in per.items()}, flush=True)

    def nb(X, Xt):
        cnt = torch.zeros(3, L * L)
        for k in range(3):
            cnt[k] = X[y == k].float().sum(0)
        logp = torch.log((cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True))
        return torch.log_softmax(Xt.float() @ logp.T, 1)
    Xs, Xst = bag([items[i]["text"] for i in tr]), bag([items[i]["text"] for i in te])
    Xf, Xft = fbag(tr), fbag(te)
    report("数えた表（文字の袋、実験19）", nb(Xs, Xst).argmax(1))
    report("数えた表（構造の手がかりだけ）", nb(Xf, Xft).argmax(1))
    NB2 = nb(Xs + Xf, Xst + Xft)
    report("数えた表（文字の袋 + 構造）", NB2.argmax(1))
    R = torch.tensor([rule_scores(items[i]["text"]) for i in te])
    half = len(yt) // 2
    w = max((0.0, 0.25, 0.5, 1.0), key=lambda w: ((NB2[:half] + w * R[:half]).argmax(1) == yt[:half]).float().mean().item())
    report("数えた表（文字 + 構造）+ ①", (NB2 + w * R).argmax(1), dict(weight=w))
    dev = torch.device(a.device)
    net = p2.Net2D(L, 3, ["xy"] * 16, patches=16, seed=a.seed).to(dev)
    p2.train(net, Xs + Xf, y, steps=a.steps, lr=0.01, batch_size=256, seed=a.seed, to_input=to_field, schedule="cosine")
    net.eval()
    Xall = Xst + Xft
    with torch.no_grad():
        S = torch.cat([net(to_field(Xall[i:i + 2000]).to(dev)).cpu() for i in range(0, len(Xall), 2000)])
    report("光の 2DWIG（文字 + 構造）", S.argmax(1))
    rec = dict(exp="21", **vars(a), n_train=len(tr), n_test=len(te), runs=runs)
    with open(os.path.join(PRIV, "results", "exp21.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dump", "parse", "eval"], required=True)
    ap.add_argument("--max_per_class", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dict(dump=dump, parse=parse, eval=evaluate)[a.stage](a)
