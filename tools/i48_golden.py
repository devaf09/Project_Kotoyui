"""I48 の答え合わせ（整理の前後で数字が変わらない事を確かめる）。

小さめの学び（既定 5,000 文）・試し（既定 200 文）で、次の数字を出す。
  - 土台: AK.Model（R1・R2・R3）の I48ao の数え
  - 単位: I48ap の UModel（型 C、自作 70 文の単位）の数え ＋ と の人・人以外
  - 驚き: I48at (二) の −log p の数え ＋ 筋の無い共振
  - その他: G.OTHER = True にした時の数え
--out で json に書き、--check で前に書いた json と比べる（小数は 1e-9 まで同じなら同じ）。
python tools/i48_golden.py --conllu ... --content ... --func ... --out golden.json
python tools/i48_golden.py --conllu ... --content ... --func ... --check golden.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_surprise as AW  # noqa: E402


def same(a, b, path=""):
    """違う所の一覧。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: 片方に無い")
            else:
                out += same(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: 長さ {len(a)} ≠ {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if isinstance(a, float) or isinstance(b, float):
        return [] if a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9 else [f"{path}: {a} ≠ {b}"]
    return [] if a == b else [f"{path}: {a} ≠ {b}"]


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--out")
    ap.add_argument("--check")
    ap.add_argument("--train", type=int, default=5000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--D", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    res = dict(学びの文=len(tr), 試しの文=len(te))
    G.OTHER = False
    m = AK.Model(S, fill, "架空・前").fit(tr, hold)
    res["土台"] = AO.measure(m, te, surf)
    res["土台"]["β"] = {k: float(v) for k, v in m.beta.items()}
    u = AP.UModel(S, fill, "架空・前").fit(tr, hold).init_units("C")
    for it in AO.TEACH_ALL:
        u.teach_unit(it)
    u.refresh()
    res["単位"] = AP.measure(u, te, surf)
    res["単位"]["単位の数"] = len(u.units)
    w = AW.WModel(S, fill, "架空・前")
    info = w.fit_weighted(tr, hold, "-logp")
    res["驚き"] = AO.measure(w, te, surf)
    res["驚き"]["筋の無い共振"] = AW.monitor(w, te)
    res["驚き"]["重さ"] = info
    G.OTHER = True
    m.refresh()
    res["その他"] = AO.measure(m, te, surf)
    G.OTHER = False
    res = json.loads(json.dumps(res, ensure_ascii=False))
    print(f"{time.time() - t0:.0f} 秒", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.check:
        diff = same(json.load(open(a.check, encoding="utf-8")), res)
        print("同じ" if not diff else "違う所:\n" + "\n".join(diff[:50]), flush=True)
        sys.exit(1 if diff else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
