"""実験17（非公開の材料）: 場面を絞って、光の 2DTG で文を一から作れるか（TinyStories の考え方、持ち主の選んだ 1 つ目、2026-11-03）

着想: 持ち主「単語を切り貼りして会話している感じ。大量の文章から単語と文法の自己構造化は難しいのか」。英語では、話題と語彙を絞ると
      小さなモデルでも筋の通った文を作れる（TinyStories、2023）。ことゆいの光の 2DTG（実験03・05）は直前 4 単位しか見ず、雑談全体で 27%。
      そこで、短い発言（20 字まで）に絞り、直前 8 単位（短い文ならほぼ文の頭から）を見て、文を最後まで作れるかを確かめる。
      あわせて、持ち主の案「語尾で個性を特定するなら特徴的な人物を想定」: 語尾がはっきりした話し方（お嬢様・老人・後輩・侍・俺様）を
      口調の帯（実験03 の口調のスイッチ）で切り替えられるかを見る。
方法: 材料 = build_persona_pairs.py の組の発言（RealPersonaChat + 持ち主が集めた Web 小説の会話文）のうち 2〜20 字、重なりは 1 回。
      口調の類 = 語尾の印（お嬢様: ですわ・わよ…、老人: のじゃ…、後輩: っす…、侍: ござる…、俺様: だぜ・じゃねえ…）、ほかは 丁寧（です・ます）か くだけた。
      SentencePiece 2048（教えの発言で学ぶ）。位置ごとに 直前 8 単位 + 口調の帯 → 次の単位。
      光: Text4f（拡散 → 4f 系 6 段 × 幅 8 + 平方根 → 窓 256 → 符号読み、実験03 の光路 1 本）。比べる: GRU（埋め込み 128・256）、口調ごとの 3-gram。
測る: 次の単位の正解率（試し 3 万位置）、作った文（口調ごとに 15 文、計 105 文）の口調の当たり（語尾の印）と、
      ローカル AI（Ollama）の「日本語の文として自然か」の評点（1〜5）。
結果: ../results/exp17.txt、例と評点 ../docs/exp17_report.md（非公開）
実行（Project_Kotoyui の直下で）: python experiments/exp17_short_gen.py --device cuda --ai=<モデル名>
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
    rec = dict(exp="17", **{k: v for k, v in vars(a).items()}, n_ctx=N_CTX, V=V, n_train=len(y), runs={})
    models = {}
    m1 = textlight.Text4f(N_CTX + 1, V + len(STYLES), V, 256, 6, 8, a.seed).to(dev)
    t1 = fit(m1, X, y, a.steps, dev, a.seed)
    m1.eval()
    acc, ce = evals(logp_of(m1, Xt, dev), yt)
    rec["runs"]["光の 2DTG（4f 系）"] = dict(acc=acc, ce=ce, params_light=m1.params()[0], params_digital=m1.params()[1], train_s=t1)
    models["光の 2DTG（4f 系）"] = m1
    g = GRU(V, V, len(STYLES)).to(dev)
    t2 = fit(g, X, y, a.steps, dev, a.seed)
    g.eval()
    acc, ce = evals(logp_of(g, Xt, dev), yt)
    rec["runs"]["GRU"] = dict(acc=acc, ce=ce, params_digital=sum(q.numel() for q in g.parameters()), train_s=t2)
    models["GRU"] = g
    tabs = [Tri(V) for _ in STYLES]
    for t, s in tr:
        tabs[s].add([1, 1] + enc[t] + [2])
    LPT = torch.stack([torch.from_numpy(np.log(np.maximum(tabs[int(Xt[i, 0]) - V].prob(int(Xt[i, -2]), int(Xt[i, -1])), 1e-12))).float()
                       for i in range(len(yt))])
    acc, ce = evals(LPT, yt)
    rec["runs"]["3-gram（口調ごと）"] = dict(acc=acc, ce=ce)
    for k, v in rec["runs"].items():
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}, flush=True)
    # 文を作る
    rng = np.random.default_rng(a.seed)
    gens = {}
    for name in list(models) + ["3-gram（口調ごと）"]:
        out = []
        for si, sty in enumerate(STYLES):
            def step(ctx, s, name=name):
                if name.startswith("3-gram"):
                    return tabs[s].prob(ctx[-2], ctx[-1])
                with torch.no_grad():
                    z = models[name](torch.tensor([[V + s] + ctx]).to(dev))
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
    with open(os.path.join(PRIV, "results", "exp17.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(PRIV, "docs", "exp17_gens.json"), "w", encoding="utf-8") as f:
        json.dump(gens, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
