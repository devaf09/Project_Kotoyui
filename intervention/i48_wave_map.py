"""行の波をビットマップにする（持ち主の頼み「行全体の波のマップ」）。D 8192、拡散の置き方（i48aj の Whole と同じ波）。

1 帯 = 1 画素（横 128 × 縦 64、左上から行ごとに帯 0, 1, 2, …）。色は HSV: 色相 = 位相（−π〜π を 0〜1）、彩度 1、
明るさ = 強さ（|X| をその絵の 99 パーセンタイルで割って 1 で切る）。差の絵は |X1 − X3| を明るさだけの灰色で。
文は持ち主の例と自作だけ。出す物:
  s1_whole / s2_whole / s3_whole : 行全体の波（who.sent(keys)）
  s1_w<k>_raw / s1_w<k>_placed    : 文 1 の語ごとの波（置く前の who.word、位置のずれ k × SHIFT を掛けた後）
  s1_gap1                         : 文 1 の、友達の後ろの間から見た波（who.gap(keys, 1)）
  diff_s1_s3                      : 文 1 と文 3 の差
  stats.json                      : 絵ごとの 強さの最大・99 パーセンタイル・「99 パーセンタイルの 1 割を超える帯の割合」
帯（B・C・D・E・G）は属性の名前を f_of で D 全体に散らすので、帯ごとのまとまった範囲は無い（属性の周波数は名前の順 × step mod D）。
python intervention/i48_wave_map.py --content ... --func ... --out <フォルダ>   （PNG を書くのに Pillow が要る）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402

W_PX, H_PX = 128, 64
SENTS = {"s1": [("昨日", "NOUN"), ("友達", "NOUN"), ("散歩", "NOUN"), ("行く", "VERB")],
         "s2": [("昨日", "NOUN"), ("公園", "NOUN"), ("犬", "NOUN"), ("散歩", "NOUN"), ("する", "VERB")],
         "s3": [("明日", "NOUN"), ("友達", "NOUN"), ("映画", "NOUN"), ("見る", "VERB")]}


def hsv_rgb(h, s, v):
    """h, s, v: 0〜1 の配列 → 0〜255 の RGB。"""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], -1) * 255 + 0.5).astype(np.uint8)


def stats(a):
    p99 = float(np.percentile(a, 99))
    return dict(最大=round(float(a.max()), 4), p99=round(p99, 4), 使う帯の割合=round(float((a > 0.1 * p99).mean()), 4))


def to_png(rgb):
    """RGB（H × W × 3、uint8）→ PNG のバイト列（標準ライブラリだけ）。"""
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(rgb.shape[0]))

    def chunk(tag, data):
        return len(data).to_bytes(4, "big") + tag + data + zlib.crc32(tag + data).to_bytes(4, "big")
    ihdr = rgb.shape[1].to_bytes(4, "big") + rgb.shape[0].to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def wave_png(X):
    a = np.abs(X)
    p99 = max(1e-12, float(np.percentile(a, 99)))
    rgb = hsv_rgb(((np.angle(X) + np.pi) / (2 * np.pi)) % 1.0, np.ones(len(X)), np.clip(a / p99, 0, 1))
    return to_png(rgb.reshape(H_PX, W_PX, 3)), stats(a)


def gray_png(a):
    p99 = max(1e-12, float(np.percentile(a, 99)))
    v = (np.clip(a / p99, 0, 1) * 255 + 0.5).astype(np.uint8)
    return to_png(np.repeat(v.reshape(H_PX, W_PX, 1), 3, axis=2)), stats(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, W_PX * H_PX, bands=("B", "C", "D", "E", "G"))
    who = AJ.Whole(S, 1)
    st = {}

    def put(name, png_stat):
        png, s = png_stat
        open(os.path.join(a.out, name + ".png"), "wb").write(png)
        st[name] = dict(s, バイト=len(png))

    X = {k: who.sent(v) for k, v in SENTS.items()}
    for k in SENTS:
        put(f"{k}_whole", wave_png(X[k]))
    for k, key in enumerate(SENTS["s1"]):
        y = who.word(*key)
        put(f"s1_w{k}_raw", wave_png(y))
        put(f"s1_w{k}_placed", wave_png(np.roll(y, k * S.shift)))
        st[f"s1_w{k}_raw"]["語"] = key[0]
        st[f"s1_w{k}_raw"]["属性の数"] = int(len(S.attrs(*key)[0]))
    put("s1_gap1", wave_png(who.gap(SENTS["s1"], 1)))
    put("diff_s1_s3", gray_png(np.abs(X["s1"] - X["s3"])))
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
