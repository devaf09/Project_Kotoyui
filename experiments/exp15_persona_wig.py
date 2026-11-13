"""実験15（非公開の材料）: 光の 2DWIG で、返事の候補の「話し手の一人称の類」と「丁寧さ」を判定する（持ち主の選んだ a、2026-11-03）

着想: 実験14 で、人格の違いは話の中身ではなく一人称・語尾などの表面に出ると分かった。返事の候補の 9 割には一人称が無いので、
      一人称を隠した語尾・言い回しから「どの人格が言いそうか」を光で当てれば、一人称の無い返事も人格の候補にできる。
方法: 実験01 の光の 2DWIG（語尾 1〜3 文字 + 文字の 2 つ組の袋を 4096 画素に散らし、64 × 64 の光の場に。Net2D の板 16 枚・区画 16、窓 = 類 1 つ）。
      persona: 返事の一人称の類（私・僕・俺・あたし・わし）。一人称の表記は「＊」で隠す。類ごとに教えを最大 --per_class 件にそろえる
      polite : 丁寧さ（です・ます、kotoyui.analyze の判定 = 答えの定義）。一人称を隠す
      比べる: 同じ袋の数えた表（ナイーブベイズ）。
材料: build_persona_pairs.py の返事（RealPersonaChat + 持ち主が集めた Web 小説の会話文。作品・会話ごとの教え・試しの分け方のまま）。
保存: 学習した光の 2DWIG は ../models/wig_<task>.pt（非公開。kotoyui の --light で使う）。
結果: ../results/exp15.txt
実行（Project_Kotoyui の直下で）: python experiments/exp15_persona_wig.py --task persona --device cuda
"""
import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict

import torch

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
sys.path.insert(0, os.path.join(KOTO, "experiments"))
import kotoyui as K  # noqa: E402
from exp01_wig_bag import L, bag, p2, to_field  # noqa: E402

PAIRS = os.path.join(PRIV, "data", "persona", "pairs.jsonl")
CLASSES = ["私", "僕", "俺", "あたし", "わし"]
HIDE = re.compile(r"(私|わたし|僕|ぼく|ボク|俺|おれ|オレ|あたし|アタシ|わし|儂)")


def hide(t):
    return HIDE.sub("＊", t)


def nb(Xtr, ytr, Xte, C, alpha=0.5):
    """数えた表（多項ナイーブベイズ）。"""
    cnt = torch.zeros(C, L * L)
    for c in range(C):
        cnt[c] = Xtr[ytr == c].float().sum(0)
    logp = torch.log((cnt + alpha) / (cnt + alpha).sum(1, keepdim=True))
    prior = torch.log(torch.bincount(ytr, minlength=C).float() / len(ytr))
    return Xte.float() @ logp.T + prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="persona", choices=["persona", "polite"])
    ap.add_argument("--per_class", type=int, default=40000)
    ap.add_argument("--max_test", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(PRIV, "results", "exp15.txt"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
    seen = set()
    tr, te = defaultdict(list), defaultdict(list)
    for r in rows:
        t = r["reply"]
        if t in seen or not (2 <= len(t) <= 120):
            continue
        seen.add(t)
        if a.task == "persona":
            if r["cls"] not in CLASSES:
                continue
            lab = CLASSES.index(r["cls"])
        else:
            lab = int(K.analyze(t)[1])
        (tr if r["split"] == "train" else te)[lab].append(hide(t))
    classes = CLASSES if a.task == "persona" else ["くだけた", "丁寧"]
    C = len(classes)
    trs, tes = [], []
    for c in range(C):
        rng.shuffle(tr[c])
        rng.shuffle(te[c])
        trs += [(t, c) for t in tr[c][:a.per_class]]
        tes += [(t, c) for t in te[c]]
    rng.shuffle(tes)
    tes = tes[:a.max_test]
    rng.shuffle(trs)
    X = bag([t for t, _ in trs])
    y = torch.tensor([c for _, c in trs])
    Xt = bag([t for t, _ in tes])
    yt = torch.tensor([c for _, c in tes])
    print(f"{a.task}: 教え {len(y)}（類ごと {torch.bincount(y, minlength=C).tolist()}）・試し {len(yt)}（{torch.bincount(yt, minlength=C).tolist()}）{time.perf_counter() - t0:.0f} 秒", flush=True)

    def report(name, S, extra=None):
        pred = S.argmax(1)
        per = [100.0 * (pred[yt == c] == c).float().mean().item() if (yt == c).any() else None for c in range(C)]
        r = dict(method=name, acc=100.0 * (pred == yt).float().mean().item(), per_class=dict(zip(classes, per)),
                 macro=sum(p for p in per if p is not None) / sum(p is not None for p in per), **(extra or {}))
        print(name, round(r["acc"], 2), "類ごとの平均", round(r["macro"], 2), {k: round(v, 1) for k, v in r["per_class"].items() if v is not None}, flush=True)
        return r
    runs = [report("数えた表（ナイーブベイズ）", nb(X, y, Xt, C))]
    dev = torch.device(a.device)
    net = p2.Net2D(L, C, ["xy"] * 16, patches=16, seed=a.seed).to(dev)
    t1 = time.perf_counter()
    p2.train(net, X, y, steps=a.steps, lr=0.01, batch_size=256, seed=a.seed, to_input=to_field, schedule="cosine")
    train_s = time.perf_counter() - t1
    net.eval()
    with torch.no_grad():
        S = torch.cat([net(to_field(Xt[i:i + 2000]).to(dev)).cpu() for i in range(0, len(Xt), 2000)])
    light = sum(q.numel() for n_, q in net.named_parameters() if n_ not in ("bias", "gain"))
    runs.append(report("光の 2DWIG", S, dict(train_s=train_s, params_light=light, params_digital=sum(q.numel() for q in net.parameters()) - light)))
    os.makedirs(os.path.join(PRIV, "models"), exist_ok=True)
    torch.save(dict(state=net.state_dict(), classes=classes, task=a.task, layers=16, patches=16, seed=a.seed),
               os.path.join(PRIV, "models", f"wig_{a.task}.pt"))
    rec = dict(exp="15", **{k: v for k, v in vars(a).items() if k != "out"}, classes=classes, n_train=len(y), n_test=len(yt),
               runs=runs, total_s=time.perf_counter() - t0, version=K.__version__, torch=torch.__version__)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
