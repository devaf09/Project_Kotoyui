"""実験18（非公開の材料）: 光の 2DTG を大きくすると、文を作る力と語尾の個性はどこまで伸びるか（持ち主「大きさが足りないなら大きくするしかない」2026-11-03）

材料・問・測り方は実験17 と同じ（20 字までの発言、直前 8 単位 + 口調の帯、試し 3 万位置、口調ごとに 15 文、語尾の当たり）。
大きさの梯子（--kind・--stages・--width・--G・--steps）:
  text4f  光路 1 本（実験17 は 6 段 × 幅 8、9000 ステップ）。段・幅・ステップを増やす
  split4f 分担の 2DTG（直前の単位の番号の余りで担当 G 個に分ける。中身は 4f 系）。器の数で大きくする
  gru     比べる: GRU（隠れ層 --hidden）
--ft_steps > 0: 人格ごとの器（持ち主の案「分けて足す」、実験17 の結論）。学習した光路を、口調ごとにその口調の位置だけで
  さらに学ばせた写しを作り、その口調の文はその写しで作る（sincnn3 の「似た器から写して学ぶ」と同じ形）。正解率は元の器で測る。
語尾の判定は実験17 の印に、文末の半角の ? ! ♪ も含めるよう直した。
結果: ../results/exp18.txt、作った文 ../docs/exp18_gens_<tag>.json（非公開）
実行（Project_Kotoyui の直下で）: python experiments/exp18_scale_gen.py --kind text4f --stages 8 --width 16 --steps 18000 --tag t8x16
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
sys.path.insert(0, os.path.join(KOTO, "experiments"))
sys.path.insert(0, os.path.join(KOTO, "tools"))
import kotoyui as K  # noqa: E402
import est_flavor_counts as efc  # noqa: E402
import textlight  # noqa: E402
from exp03_4f import fit, logp_of, evals  # noqa: E402

PAIRS = os.path.join(PRIV, "data", "persona", "pairs.jsonl")
N_CTX = 8
STY = [("お嬢様", r"(ですわ|ますわ|ですのよ|ますのよ|でしてよ|ですの|わよ|かしら)[。！？…]*$"),
       ("老人", r"(のじゃ|じゃのう|じゃよ|おるのう|じゃわい)[。！？…]*$"),
       ("後輩", r"(っす|っすね|っすよ|っすか)[。！？…]*$"),
       ("侍", r"(ござる|ござらん|でござる|候)[。！？…]*$"),
       ("俺様", r"(だぜ|ぜ|だろうが|じゃねえ|じゃねぇ|ねぇか|ねえか)[。！？…]*$")]
STY = [(n, p.replace("[。！？…]*$", "[。！？…!?♪\\s]*$")) for n, p in STY]
STYLES = ["丁寧", "くだけた"] + [n for n, _ in STY]


def style_of(t):
    for n, p in STY:
        if re.search(p, t):
            return n
    return "丁寧" if K.analyze(t)[1] else "くだけた"


class GRU(torch.nn.Module):
    def __init__(self, V, C, n_tag):
        super().__init__()
        self.emb = torch.nn.Embedding(V + n_tag, 128)
        self.rnn = torch.nn.GRU(128, 256, batch_first=True)
        self.out = torch.nn.Linear(256, C)

    def forward(self, X):
        return self.out(self.rnn(self.emb(X))[0][:, -1])


class Tri:
    """口調ごとの 3-gram（足し算のなめらかさ 0.1 と 2-gram・1-gram への後退を混ぜる）。"""

    def __init__(self, V):
        self.V, self.c3, self.c2, self.c1 = V, {}, {}, np.zeros(V)

    def add(self, seq):
        for i in range(2, len(seq)):
            a, b, c = seq[i - 2], seq[i - 1], seq[i]
            self.c3.setdefault((a, b), {}).setdefault(c, 0)
            self.c3[(a, b)][c] += 1
            self.c2.setdefault(b, {}).setdefault(c, 0)
            self.c2[b][c] += 1
            self.c1[c] += 1

    def prob(self, a, b):
        p = 0.1 * (self.c1 + 0.1) / (self.c1.sum() + 0.1 * self.V)
        d2 = self.c2.get(b)
        if d2:
            v = np.zeros(self.V)
            for k, n in d2.items():
                v[k] = n
            p = p + 0.3 * v / v.sum()
        d3 = self.c3.get((a, b))
        if d3:
            v = np.zeros(self.V)
            for k, n in d3.items():
                v[k] = n
            p = p + 0.6 * v / v.sum()
        return p / p.sum()


def generate(step_probs, style, rng, topk=20, max_len=30):
    seq = [1] * N_CTX
    for _ in range(max_len):
        p = step_probs(seq[-N_CTX:], style).copy()
        p[[0, 1]] = 0
        if seq[-1] == seq[-2] and seq[-1] > 2:
            p[seq[-1]] = 0
        top = np.argsort(-p)[:topk]
        y = int(rng.choice(top, p=p[top] / p[top].sum()))
        if y == 2:
            return seq[N_CTX:], True
        seq.append(y)
    return seq[N_CTX:], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--kind", default="text4f", choices=["text4f", "split4f", "gru"])
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--ft_steps", type=int, default=0, help="人格ごとの器: 口調ごとに写して学ぶステップ数（text4f だけ）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_train", type=int, default=3000000)
    ap.add_argument("--n_gen", type=int, default=15, help="口調ごとに作る文の数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ai", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    t0 = time.perf_counter()
    seen, tr, te = set(), [], []
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        for t in (r["prev"], r["reply"]):
            if t in seen or not (2 <= len(t) <= 20):
                continue
            seen.add(t)
            (tr if r["split"] == "train" else te).append((t, STYLES.index(style_of(t))))
    print(f"発言 教え {len(tr)}・試し {len(te)}（{time.perf_counter() - t0:.0f} 秒）、口調ごと {np.bincount([s for _, s in tr], minlength=len(STYLES)).tolist()}", flush=True)
    efc.SPM_DIR = os.path.join(PRIV, "data", "persona")                         # SentencePiece は非公開の置き場に
    sp = efc.train_spm([t for t, _ in tr], 2048, "short17")
    V = sp.get_piece_size()
    enc = {t: sp.encode(t) for t, _ in tr + te}

    def pos(items, cap, seed):
        X, y = [], []
        for t, s in items:
            seq = [1] * N_CTX + enc[t] + [2]
            for i in range(N_CTX, len(seq)):
                X.append([V + s] + seq[i - N_CTX:i])
                y.append(seq[i])
        X, y = torch.tensor(X), torch.tensor(y)
        if len(y) > cap:
            i = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))[:cap]
            X, y = X[i], y[i]
        return X, y
    X, y = pos(tr, a.max_train, a.seed)
    Xt, yt = pos(te, 30000, 0)
    print(f"位置 教え {len(y)}・試し {len(yt)}（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    rec = dict(exp="18", **{k: v for k, v in vars(a).items()}, n_ctx=N_CTX, V=V, n_train=len(y), runs={})
    t1 = time.perf_counter()
    if a.kind == "text4f":
        m = textlight.Text4f(N_CTX + 1, V + len(STYLES), V, 256, a.stages, a.width, a.seed).to(dev)
        fit(m, X, y, a.steps, dev, a.seed)
        size = dict(params_light=m.params()[0], params_digital=m.params()[1])
        fn = m
    elif a.kind == "split4f":
        m = textlight.Split4f(N_CTX + 1, V + len(STYLES), V, a.G, 256, a.stages, a.width, a.seed).to(dev)
        textlight.train_split(m, X, y, a.steps, seed=a.seed, dev=dev, log=lambda msg: print(msg, flush=True))
        lp, dg = m.params()
        size = dict(params_light=lp, params_digital=dg)
        fn = m.logits
    else:
        m = GRU(V, V, len(STYLES)).to(dev)
        if a.hidden != 256:
            m.rnn = torch.nn.GRU(128, a.hidden, batch_first=True).to(dev)
            m.out = torch.nn.Linear(a.hidden, V).to(dev)
        fit(m, X, y, a.steps, dev, a.seed)
        size = dict(params_digital=sum(q.numel() for q in m.parameters()))
        fn = m
    m.eval()
    train_s = time.perf_counter() - t1
    acc, ce = evals(logp_of(fn, Xt, dev), yt)
    name = a.tag or a.kind
    rec["runs"][name] = dict(acc=acc, ce=ce, train_s=train_s, **size)
    print(name, rec["runs"][name], flush=True)
    per_style = {}
    if a.ft_steps and a.kind == "text4f":                                     # 人格ごとの器: 口調ごとに写して、その口調の位置だけでさらに学ぶ
        import copy
        for si, sty in enumerate(STYLES):
            own = (X[:, 0] == V + si).nonzero().squeeze(1)
            mc = copy.deepcopy(m).train()
            fit(mc, X[own], y[own], a.ft_steps, dev, a.seed + si)
            mc.eval()
            per_style[si] = mc
            print(f"人格の器 {sty}: 教え {len(own)} 位置", flush=True)
    models = {name: fn}
    # 文を作る
    rng = np.random.default_rng(a.seed)
    gens = {}
    for name in list(models):
        out = []
        for si, sty in enumerate(STYLES):
            def step(ctx, s, name=name):
                with torch.no_grad():
                    z = per_style.get(s, models[name])(torch.tensor([[V + s] + ctx]).to(dev))
                return F.softmax(z.float(), 1)[0].cpu().numpy()
            for _ in range(a.n_gen):
                ids, closed = generate(step, si, rng)
                text = sp.decode(ids)
                out.append(dict(style=sty, text=text, closed=closed, hit=style_of(text) == sty, copied=text in seen))
        gens[name] = out
        rec["runs"].setdefault(name, {})
        rec["runs"][name]["style_hit"] = sum(o["hit"] for o in out) / len(out)
        rec["runs"][name]["copied"] = sum(o["copied"] for o in out) / len(out)
        print(name, "口調の当たり", round(rec["runs"][name]["style_hit"], 3), "材料と同じ文", round(rec["runs"][name]["copied"], 3), flush=True)
    if a.ai:
        from ollama_client import Ollama
        ai = Ollama(a.ai, timeout=300)
        for name, out in gens.items():
            sc = []
            for o in out:
                q = f"次の文は、日本語の会話の 1 文として自然ですか。1（意味が通らない）〜5（とても自然）の整数で、1 行目に数字だけ、2 行目に短い理由を書いてください。\n\n「{o['text']}」"
                try:
                    r = ai._post([dict(role="user", content=q)], temperature=0.0)
                    m = re.search(r"[1-5]", r)
                    o["ai_score"], o["ai_reason"] = (int(m.group()) if m else None), r.split("\n", 1)[-1].strip()
                except Exception as e:
                    o["ai_score"], o["ai_reason"] = None, str(e)
                if o["ai_score"]:
                    sc.append(o["ai_score"])
            rec["runs"][name]["ai_mean"] = sum(sc) / max(1, len(sc))
            rec["runs"][name]["ai_ge4"] = sum(s >= 4 for s in sc) / max(1, len(sc))
            print(name, "AI の平均", round(rec["runs"][name]["ai_mean"], 2), "4 以上", round(rec["runs"][name]["ai_ge4"], 2), flush=True)
    rec["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.join(PRIV, "results"), exist_ok=True)
    with open(os.path.join(PRIV, "results", "exp18.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(PRIV, "docs", f"exp18_gens_{a.tag or a.kind}.json"), "w", encoding="utf-8") as f:
        json.dump(gens, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
