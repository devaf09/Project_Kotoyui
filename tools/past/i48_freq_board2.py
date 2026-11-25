"""周波数を紐付けた語板（tools/past/i48_freq_board.py）の続き（持ち主の案、未確定）。

1. 1,000 倍の周波数のまま学習して、ダウンコンバートを最後にやる:
   (a) 本物の波で確かめる: 語の値を、点を並べた実数の波（1 歩 = 2,048 点、窓 200 歩）にする。帯 b の搬送は 1,000 + b + ν_j。
       学習 = 相手の語の波を距離だけ遅らせて足す（1,000 倍の帯のまま）。読み = 覚えの波と候補の波を掛けて平均を取る
       （掛け算と低域通過 = 最後のダウンコンバート）。直交の波（90 度ずらした候補）も掛けて、包絡（I・Q の大きさ）でも読む。
       位相の式で出した共振と同じになるか、と、学習した文の語の戻り（読む位置のずれ 0・1/4,000 歩・0.5 歩）。
   (b) 学習の時の位置の揺れ: 組ごとに距離に揺れ（標準偏差 σ 歩）を入れて覚える。先に下げる（ずれ ν だけで覚える）と、
       最後に下げる（1,000 倍の搬送のまま覚え、読む所で実部 ／ 包絡）を比べる。
2. フィルターと増幅:
   覚え方（フィルター）: 組の 2 語の帯ごとの値の近さ c_b（0〜1）で、帯ごとに覚える強さを変える
     被りを強く = 1 + α c_b、被りだけ = c_b、被らない帯を強く = 1 + α (1 − c_b)（α = 1・4）
   読み方（増幅）: 帯ごとの共振を、候補の間で平均 0・ばらつき 1 にそろえて（自動の利得）足す、そろえた値の 0 より上だけ 2 乗して足す
3. 周波数で抜き出す案:
   山の形に名前の位相を混ぜる（ε = 0.1・0.2・0.3）、重ねる（10 文を乱数の鍵 ／ 周波数の席を分けて 1 本に）、
   ずれ探し（前回の 2 段 ／ 全部の周波数で細かく総当たり）
手で置く物: 点の数（1 歩 2,048 点・窓 200 歩）、σ の並び、α の並び、ε の並び、重ねる文の数 10、席の数 10。
python tools/past/i48_freq_board2.py --content .../content_boards.json --out out.json
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48_freq_board as FB  # noqa: E402

NB, NC = len(FB.BANDS), FB.NC
FS, TW = 2048, 200                 # 1 歩の点の数、窓の歩数（ν の間隔 0.005 が窓にちょうど 1 周期で入る）
JIT = (0.0, 1e-5, 1e-4, 2.5e-4, 1e-3, 1e-2, 0.1)
ALPHAS = (1.0, 4.0)
EPSS = (0.1, 0.2, 0.3)
W = 0.5


class Board2(FB.Board):
    def __init__(self, *a, eps=0.0, band_carrier=False, **k):
        super().__init__(*a, **k)
        if band_carrier:                                          # 帯ごとに搬送を 1 ずつずらす（1 本の実数の波に全部の帯を載せるため）
            self.f = self.nu + FB.F0 + np.repeat(np.arange(NB), NC)
        if eps > 0:
            for (b, a_), s in list(self.shapes.items()):
                r = np.random.default_rng(zlib.crc32(("名:" + b + "|" + a_).encode()))
                nm = np.exp(2j * np.pi * r.random(NC)) / np.sqrt(NC)
                self.shapes[(b, a_)] = (1 - eps) * s + eps * nm
            self.S = np.stack([self.value(self.att[w]) for w in self.lemmas])

    def band_cos(self, i, k):
        a = self.S[i].reshape(NB, NC)
        b = self.S[k].reshape(NB, NC)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        c = np.real(np.sum(np.conj(a) * b, axis=1)) / np.maximum(1e-12, na * nb)
        return np.clip(c, 0, 1) * ((na > 0) & (nb > 0))

    def learn2(self, lines, filt=None, alpha=1.0, jitter=0.0, f=None, rng=None):
        f = self.f if f is None else f
        R = np.zeros_like(self.S)
        for ln in lines:
            ids = [self.li[w] for w in ln]
            for p, i in enumerate(ids):
                for q, k in enumerate(ids):
                    if q == p:
                        continue
                    d = q - p + (rng.normal(0, jitter) if jitter > 0 else 0.0)
                    g = 1.0
                    if filt:
                        c = self.band_cos(i, k)
                        g = {"被りを強く": 1 + alpha * c, "被りだけ": c, "被らない帯を強く": 1 + alpha * (1 - c)}[filt]
                        g = np.repeat(g, NC)
                    R[i] += g * self.S[k] * np.exp(-2j * np.pi * f * d)
        return R

    def band_res(self, R, ids, p, delta=0.0, f=None):
        f = self.f if f is None else f
        Q = np.zeros(self.S.shape[1], dtype=np.complex128)
        for pi, i in enumerate(ids):
            if pi != p:
                Q += np.conj(R[i]) * np.exp(-2j * np.pi * f * (p - pi + delta))
        return (self.S * Q[None, :]).reshape(len(self.S), NB, NC).sum(2)        # 候補 × 帯（複素）


def amp(Z, kind):
    if kind == "そのまま":
        return np.real(Z).sum(1)
    if kind == "包絡":
        return np.abs(Z).sum(1)
    X = np.real(Z)
    z = (X - X.mean(0)) / np.maximum(1e-12, X.std(0))
    if kind == "自動の利得":
        return z.sum(1)
    if kind == "0より上を2乗":
        return (np.maximum(0, z) ** 2).sum(1)
    raise ValueError(kind)


def evalset(bd, R, lines, kind, Rn=None, delta=0.0, f=None):
    h = h5 = n = 0
    z, above = [], []
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        if len(ids) < 2:
            continue
        for p, t in enumerate(ids):
            sc = amp(bd.band_res(R, ids, p, delta, f), kind)
            r = int((sc > sc[t]).sum())
            h += r == 0
            h5 += r < 5
            n += 1
            if Rn is not None:
                nz = np.array([amp(bd.band_res(Rk, ids, p, delta, f), kind)[t] for Rk in Rn])
                z.append((sc[t] - nz.mean()) / max(1e-12, nz.std()))
                above.append(sc[t] > np.quantile(nz, 0.95))
    out = dict(一位=round(h / n, 4), 上位5=round(h5 / n, 4))
    if Rn is not None:
        out.update(ノイズからの離れ_z=round(float(np.mean(z)), 2), ノイズの上位5パーセントより上=round(float(np.mean(above)), 4))
    return out


# ---- 1(a) 本物の波 ----
def rf_spec(bd, val, delay):
    """語の値を、遅れ delay の実数の波のスペクトル（rfft の並び）にする。"""
    N = FS * TW
    X = np.zeros(N // 2 + 1, dtype=np.complex128)
    k = np.rint(bd.f * TW).astype(np.int64)
    np.add.at(X, k, val * np.exp(-2j * np.pi * bd.f * delay) * (N / 2))
    return X


def rf_check(bd, lines, deltas=(0.0, 0.00025, 0.5)):
    N = FS * TW
    assert np.allclose(bd.f * TW, np.rint(bd.f * TW)) and np.rint(bd.f * TW).max() < N // 2
    C = np.stack([np.fft.irfft(rf_spec(bd, bd.S[c], 0.0), N).astype(np.float32) for c in range(len(bd.S))])
    Cq = np.stack([np.fft.irfft(rf_spec(bd, -1j * bd.S[c], 0.0), N).astype(np.float32) for c in range(len(bd.S))])
    out = {}
    worst = 0.0
    used = sorted({bd.li[w] for s in lines for w in s})
    # 覚えの波（1,000 倍の帯のまま、実数の波を遅らせて足す）
    Rspec = {i: np.zeros(N // 2 + 1, dtype=np.complex128) for i in used}
    for s in lines:
        ids = [bd.li[w] for w in s]
        for p, i in enumerate(ids):
            for q, k in enumerate(ids):
                if q != p:
                    Rspec[i] += rf_spec(bd, bd.S[k], q - p)
    R = bd.learn2(lines)
    for dl in deltas:
        h = he = hp = n = 0
        for s in lines:
            ids = [bd.li[w] for w in s]
            for p, t in enumerate(ids):
                Qs = np.zeros(N // 2 + 1, dtype=np.complex128)
                for pi, i in enumerate(ids):
                    if pi != p:
                        Qs += Rspec[i] * np.exp(2j * np.pi * np.arange(N // 2 + 1) / TW * (p - pi + dl))   # 覚えの波を進める
                Qt = np.fft.irfft(Qs, N).astype(np.float32)
                I = (C @ Qt) / N                                   # 掛けて平均（低域通過）= 最後のダウンコンバート
                Qq = (Cq @ Qt) / N
                ph = np.real(bd.band_res(R, ids, p, dl)).sum(1) / 2
                worst = max(worst, float(np.max(np.abs(I - ph)) / max(1e-12, np.max(np.abs(ph)))))
                h += int(np.argmax(I) == t)
                he += int(np.argmax(np.hypot(I, Qq)) == t)
                hp += int(np.argmax(ph) == t)
                n += 1
        out[f"読む位置のずれ {dl}"] = dict(本物の波_掛けて平均_一位=round(h / n, 4), 本物の波_包絡_一位=round(he / n, 4),
                                    位相の式_一位=round(hp / n, 4))
    out["本物の波と位相の式の差_最大(割合)"] = float(f"{worst:.2e}")
    out["行"] = len(lines)
    out["点の数"] = N
    return out


def stack_test(bd, lines, M=10, seats=10, rng=None):
    """10 文を 1 本に重ねて戻す: 乱数の位相の鍵 ／ 周波数の席（帯の中の周波数を 10 の席に分け、文ごとに 1 席だけ使う）。"""
    L = lines[:M]
    out = {}
    keys = np.exp(2j * np.pi * rng.random((M, NB * NC)))
    X = sum(keys[m] * FB.line_wave(bd, [bd.li[w] for w in s], 0.0) for m, s in enumerate(L))
    h = n = 0
    for m, s in enumerate(L):
        ids = [bd.li[w] for w in s]
        g = FB.read_wave(bd, np.conj(keys[m]) * X, len(ids), 0.0, env=False).argmax(1)
        h += int((g == np.array(ids)).sum())
        n += len(ids)
    out["乱数の鍵"] = round(h / n, 4)
    seat = np.tile(np.arange(NC) % seats, NB)
    X = np.zeros(NB * NC, dtype=np.complex128)
    for m, s in enumerate(L):
        X += (seat == m % seats) * FB.line_wave(bd, [bd.li[w] for w in s], 0.0)
    h = n = 0
    for m, s in enumerate(L):
        ids = [bd.li[w] for w in s]
        mask = seat == m % seats
        Sm = bd.S * mask[None, :]
        Sm = Sm / np.maximum(1e-12, np.linalg.norm(Sm, axis=1, keepdims=True))
        for p, t in enumerate(ids):
            Y = (X * mask) * np.conj(bd.ph(p))
            h += int(np.argmax(np.real(np.conj(Sm) @ Y)) == t)
            n += 1
    out["周波数の席"] = round(h / n, 4)
    out["重ねた文"] = M
    return out


def search_test(bd, lines, rng):
    lt = [s for s in lines if len(s) >= 2]
    taus = rng.uniform(-1.5, 1.5, len(lt))
    grid = np.arange(-2, 2.0001, 0.01)
    errs, h, n = [], 0, 0
    for s, tau in zip(lt, taus):
        ids = [bd.li[w] for w in s]
        X = FB.line_wave(bd, ids, tau)
        g = [FB.read_wave(bd, X, len(ids), t, env=False).max(1).sum() for t in grid]
        th = grid[int(np.argmax(g))]
        errs.append(abs(th - tau))
        gg = FB.read_wave(bd, X, len(ids), th, env=False).argmax(1)
        h += int((gg == np.array(ids)).sum())
        n += len(ids)
    return dict(語の戻り=round(h / n, 4), ずれの読み違い_平均=round(float(np.mean(errs)), 4),
                ずれの読み違い_中央=round(float(np.median(errs)), 4),
                外した行_0点05歩より=round(float(np.mean(np.array(errs) > 0.05)), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rf_lines", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    cb_all = {}
    for w in json.load(open(a.content, encoding="utf-8")):
        if w["name"] not in cb_all or w.get("freq", 0) > cb_all[w["name"]].get("freq", 0):
            cb_all[w["name"]] = w
    train = [s.split() for _, s in FB.LINES]
    ok = [s.split() for _, s in FB.UNSEEN_OK]
    bad = [s.split() for _, s in FB.UNSEEN_BAD]
    lemmas = sorted({w for s in train for w in s})
    rng = np.random.default_rng(7)
    flat = [w for s in train for w in s]
    noise_lines = []
    for _ in range(FB.N_NOISE):
        perm = list(rng.permutation(flat))
        nl, k = [], 0
        for s in train:
            nl.append(perm[k:k + len(s)])
            k += len(s)
        noise_lines.append(nl)
    res = dict(語=len(lemmas), 学習の文=len(train), W=W)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 1(a) 本物の波
    t1 = time.time()
    bdr = Board2(cb_all, lemmas, "乱数", None, W, band_carrier=True)
    rl = [s for s in train if len(s) >= 3][: a.rf_lines]
    res["1a_本物の波で最後にダウンコンバート"] = rf_check(bdr, rl)
    res["1a_本物の波で最後にダウンコンバート"]["秒"] = round(time.time() - t1)
    print("1a", json.dumps(res["1a_本物の波で最後にダウンコンバート"], ensure_ascii=False), flush=True)
    del bdr
    save()

    # 1(b) 学習の時の位置の揺れ
    t1 = time.time()
    bd = Board2(cb_all, lemmas, "乱数", None, W)
    fc = bd.nu + FB.F0
    jr = {}
    for sg in JIT:
        r_ = {}
        R0 = bd.learn2(train, jitter=sg, f=bd.nu, rng=np.random.default_rng(3))
        r_["先に下げる(ν だけで覚える)"] = evalset(bd, R0, train, "そのまま", f=bd.nu)["一位"]
        Rc = bd.learn2(train, jitter=sg, f=fc, rng=np.random.default_rng(3))
        r_["最後に下げる_実部"] = evalset(bd, Rc, train, "そのまま", f=fc)["一位"]
        r_["最後に下げる_帯ごとの包絡"] = evalset(bd, Rc, train, "包絡", f=fc)["一位"]
        jr[f"揺れ σ {sg} 歩"] = r_
        print("1b", sg, r_, flush=True)
    res["1b_学習の時の位置の揺れ_学習した文の一位"] = jr
    res["1b_秒"] = round(time.time() - t1)
    save()

    # 2 フィルターと増幅、3 山に名前の位相
    t1 = time.time()
    res["2_フィルターと増幅"] = {}
    configs = [("乱数", None, 0.0), ("山 σ1", 1.0, 0.0)] + [(f"山 σ1・名前の位相 ε {e}", 1.0, e) for e in EPSS]
    for lay, sigma, eps in configs:
        bd = Board2(cb_all, lemmas, "乱数" if sigma is None else "山", sigma, W, eps=eps)
        learns = [("均一", None, 0.0)] + [(f"{fl} α{al:g}", fl, al) for fl in ("被りを強く", "被らない帯を強く") for al in ALPHAS] + [("被りだけ", "被りだけ", 0.0)]
        for ln_, fl, al in learns:
            R = bd.learn2(train, filt=fl, alpha=al)
            Rn = [bd.learn2(nl, filt=fl, alpha=al) for nl in noise_lines]
            for kind in ("そのまま", "自動の利得", "0より上を2乗"):
                nm = f"{lay}・覚え方 {ln_}・読み方 {kind}"
                res["2_フィルターと増幅"][nm] = dict(学習した文=evalset(bd, R, train, kind),
                                                 学習に無い大体正しい文=evalset(bd, R, ok, kind, Rn),
                                                 おかしな文=evalset(bd, R, bad, kind, Rn))
                print(nm, json.dumps(res["2_フィルターと増幅"][nm], ensure_ascii=False), flush=True)
            save()
        if True:
            res.setdefault("3_重ねる", {})[lay] = stack_test(bd, train, rng=np.random.default_rng(5))
            res.setdefault("3_ずれを総当たりで探す", {})[lay] = search_test(bd, train, np.random.default_rng(11))
            print(lay, res["3_重ねる"][lay], res["3_ずれを総当たりで探す"][lay], flush=True)
            save()
    res["2_秒"] = round(time.time() - t1)
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
