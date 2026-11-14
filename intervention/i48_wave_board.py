"""介入の研究 I48: 意味の波を、すくった後も周波数のまま持つ板（I47 の形を道具として作り直した物）

持ち主の方針（2026-11-13）:
  「可能な限り周波数成分として持っておいてもらいたい。今のスペックなら周波数から展開するのにそれほどオーバーヘッドが
    気にならないと思うから、容量と精度に注力したい」
  「層を増やすことも考えないといけない時が来る、早い方が変更が少ないかも」

I47b との違い:
  - すくった板を「語と明るさ」の表でなく、波のまま持つ。層ごとに、共鳴した場面の次の行の波を共鳴の強さ（の 2 乗）で
    重ね合わせた 1 本の波 S_l。層の間もこの波で渡す。語に戻す（語の波と共鳴させて読み出す）のは最後だけ。
  - そろえる（非線形）は 2 つの形を持つ。
      読み出してそろえる : 1 度語の波と共鳴させて読み出し、強い所の平均を 1 にして残し、その強さで語の波を重ね直す。
                           I47b の「語ごとにそろえる」と同じ計算（混信の分だけずれる）。
      周波数ごとにそろえる: 波の枠ごとの大きさで同じそろえ方をする。語に戻さない。語ごとにそろえるのとは別の計算。
  - 層は窓（前の何行を見るか）の並びで決まり、後から足せる（WaveBoard.add_layer）。場面は (作品, 行) で持つので、
    新しい窓の層は、しまってある行から作り直せる。
  - 層ごとに GPU を分ける（--devices cuda:0,cuda:1 で層を順に割り当てる）。載り切らない分は PC のメモリに置き、
    問いをまとめて流す（共鳴の上位を区切りごとに合わせる）。GPU の作業用のメモリは使い終わったら空ける。
  - 数の持ち方（--store）: c128（1 枠 16 バイト）、c64（8）、p8（語の波は位相だけ 8 ビット = 1、場面の波は半精度の実部・虚部 = 4）。
  - 進み具合をログに出す（--log、無ければ標準エラー）。

測る物（I47b と同じ分け方・同じ物差し）:
  当たり（上位 10 語に次の行の語が 1 つでも）、中身（よく出る 50 語を除いた再現率）、板に残る語の数、
  向き（すくった波が、100 行の候補から本当の次の行を上位 10 に入れる割合。偶然なら 0.10）、
  表と波の重なり（同じ形の表の板と波の板で、上位 10 語がいくつ同じか。混信の少なさ）、
  大きさ（場面 1 つ・1 層あたりのバイト）と時間。

  # 容量と精度の表（波の長さ D と数の持ち方を変える）
  python intervention/i48_wave_board.py --cache <parsed.json> --D 512,1024,2048,4096 --store c128,c64,p8 \
      --devices cuda:0,cuda:1 --out i48a.json --log i48a.log
  # 夜の学び直し: 昼に足した会話（chat_bot3.py --wave_day の JSONL）も学び、前の日の結果と比べて 合流／戻す を判定
  python intervention/i48_wave_board.py --cache <parsed.json> --day day.jsonl --prev prev.json --out today.json
  # 層を足す（例: 100 行の層を 5 つめに）
  python intervention/i48_wave_board.py --cache <parsed.json> --wins 1,3,10,30,100

torch が無い・GPU が無い時は numpy（と scipy があれば疎行列）で同じ答えを出す（遅い）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import torch
except Exception:                                           # torch が無ければ numpy だけで動く
    torch = None
try:
    import scipy.sparse as sp
except Exception:
    sp = None

CONTENT = ("NOUN", "PROPN", "VERB", "ADJ")
STOP = {"する", "いる", "ある", "なる", "言う", "思う", "こと", "もの", "よう", "それ", "これ", "の", "ん", "方", "人", "今", "何", "自分", "私", "俺", "僕", "君", "あなた"}
WINS = (1, 3, 10, 30)
log = logging.getLogger("i48")


def content(toks):
    return [t["lemma"] for t in toks if t["pos"] in CONTENT and t["lemma"] not in STOP]


# ---------------------------------------------------------------- 数の持ち方
class Store:
    """波のしまい方。encode でしまい、decode で計算用の複素数に戻す（使う瞬間だけ）。
    unit=True は大きさ 1 の波（語の波）で、p8 の時は位相だけ 8 ビットで持てる。"""

    NAMES = ("c128", "c64", "p8")

    def __init__(self, name, unit=False):
        assert name in self.NAMES, name
        self.name, self.unit = name, unit

    @property
    def cdtype(self):
        return np.complex128 if self.name == "c128" else np.complex64

    def bytes_per(self, D):
        return {"c128": 16, "c64": 8, "p8": 1 if self.unit else 4}[self.name] * D

    def encode(self, X):
        if self.name == "c128":
            return np.ascontiguousarray(X, dtype=np.complex128)
        if self.name == "c64":
            return np.ascontiguousarray(X, dtype=np.complex64)
        if self.unit:                                       # 位相だけ: 0〜255 = 0〜2π
            return np.mod(np.round(np.angle(X) * (256 / (2 * np.pi))).astype(np.int64), 256).astype(np.uint8)
        return np.stack([X.real, X.imag], -1).astype(np.float16)

    def decode_np(self, E):
        if self.name in ("c128", "c64"):
            return E
        if self.unit:
            return np.exp(1j * (E.astype(np.float32) * (2 * np.pi / 256))).astype(np.complex64)
        return (E[..., 0].astype(np.float32) + 1j * E[..., 1].astype(np.float32)).astype(np.complex64)

    def decode_t(self, E):
        if self.name in ("c128", "c64"):
            return E
        if self.unit:
            return torch.polar(torch.ones(E.shape, device=E.device), E.float() * (2 * math.pi / 256))
        return torch.complex(E[..., 0].float(), E[..., 1].float())


# ---------------------------------------------------------------- 置き場（GPU か PC のメモリ）
def parse_devices(s):
    if torch is None or s in ("", "numpy", "cpu"):
        return [None]
    if s == "auto":
        n = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(n)] or [None]
    return [torch.device(x) for x in s.split(",")]


def free_bytes(dev):
    if dev is None or dev.type != "cuda":
        return 0
    f, _ = torch.cuda.mem_get_info(dev)
    return f


def gpu_note(devs):
    if torch is None:
        return ""
    out = []
    for d in devs:
        if d is not None and d.type == "cuda":
            f, t = torch.cuda.mem_get_info(d)
            out.append(f"{d}: 使用 {(t - f) / 2**30:.1f}/{t / 2**30:.1f} GB")
    return "、".join(out)


def release(dev):
    if dev is not None and dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()


class Bank:
    """しまってある波（行 × D）。区切り（chunk）ごとに、載る分は GPU に、載らない分は PC のメモリに置く。
    top(Q, k): 問いの波をまとめて共鳴させ、|K q*| の強い順に k 個（番号, 強さ）。"""

    def __init__(self, X, store, dev=None, chunk=0, budget=0.8, name="", enc=None):
        self.store, self.dev, self.name = store, dev, name
        self.N, self.D = (X if enc is None else enc).shape[:2]
        chunk = chunk or max(4096, (1 << 29) // (8 * self.D))   # 計算用に戻した 1 区切りが 512 MB ほど
        self.enc = store.encode(X) if enc is None else enc  # enc: しまった形のまま受け取る（保存した板の読み込み）
        self.chunk = chunk
        self.gpu = {}
        self.host = None                                    # GPU に載らなかった区切りだけ、PC のメモリに持つ
        self._nbytes = self.enc.nbytes
        if dev is not None:
            reserve = 6 * chunk * self.D * 8                # 測る時の作業用（区切りを複素数に戻す分）を先に空けておく
            room = max(0.0, free_bytes(dev) * budget - reserve) if dev.type == "cuda" else float("inf")
            for c0 in range(0, self.N, chunk):
                part = self.enc[c0:c0 + chunk]
                if part.nbytes > room:
                    break
                self.gpu[c0] = torch.from_numpy(part).to(dev)
                room -= part.nbytes
            if self.gpu:                                    # GPU に載せた区切りの写しは PC のメモリに残さない
                self.host = {c0: self.enc[c0:c0 + chunk].copy() for c0 in range(0, self.N, chunk) if c0 not in self.gpu}
                self.enc = None
            if len(self.gpu) * chunk < self.N:
                log.info("%s: GPU に載らない分を PC のメモリから流す（%d/%d 区切りが GPU）", name, len(self.gpu),
                         (self.N + chunk - 1) // chunk)

    @property
    def nbytes(self):
        return self._nbytes

    def free(self):
        self.gpu.clear()
        release(self.dev)

    def dump(self):
        """しまった形のまま、全部を 1 つの並びで（保存用）。GPU に載せた区切りは PC のメモリに戻して集める。"""
        if self.enc is not None:
            return self.enc
        return np.concatenate([self.gpu[c0].cpu().numpy() if c0 in self.gpu else self.host[c0] for c0 in range(0, self.N, self.chunk)])

    def top(self, Q, k):
        k = min(k, self.N)
        if self.dev is None:
            Qc = np.conj(Q).T.astype(self.store.cdtype)
            best_v = best_i = None
            for c0 in range(0, self.N, self.chunk):
                R = np.abs(self.store.decode_np(self.enc[c0:c0 + self.chunk]) @ Qc)      # 区切り × 問い
                kk = min(k, len(R))
                ii = np.argpartition(-R, kk - 1, axis=0)[:kk]
                vv = np.take_along_axis(R, ii, 0)
                ii = ii + c0
                if best_v is None:
                    best_v, best_i = vv, ii
                else:
                    V, I = np.concatenate([best_v, vv]), np.concatenate([best_i, ii])
                    jj = np.argpartition(-V, k - 1, axis=0)[:k]
                    best_v, best_i = np.take_along_axis(V, jj, 0), np.take_along_axis(I, jj, 0)
            o = np.argsort(-best_v, axis=0)
            return np.take_along_axis(best_i, o, 0).T, np.take_along_axis(best_v, o, 0).T
        cdt = torch.complex128 if self.store.name == "c128" else torch.complex64
        Qt = torch.from_numpy(np.ascontiguousarray(np.conj(Q).T)).to(self.dev, cdt)
        best_v = best_i = None
        for c0 in range(0, self.N, self.chunk):
            E = self.gpu.get(c0)
            if E is None:
                part = self.host[c0] if self.host is not None else self.enc[c0:c0 + self.chunk]
                E = torch.from_numpy(part).to(self.dev, non_blocking=True)
            R = (self.store.decode_t(E).to(cdt) @ Qt).abs()                  # 区切り × 問い
            vv, ii = torch.topk(R, min(k, R.shape[0]), dim=0)
            ii = ii + c0
            if best_v is None:
                best_v, best_i = vv, ii
            else:
                V, I = torch.cat([best_v, vv]), torch.cat([best_i, ii])
                vv, jj = torch.topk(V, k, dim=0)
                best_v, best_i = vv, torch.gather(I, 0, jj)
            del R, E
        return best_i.T.cpu().numpy(), best_v.T.cpu().numpy().astype(np.float64)

    def rows(self, idx):
        """番号の波を計算用の複素数で（PC のメモリから）。GPU に載せない置き場（次の行の波）だけで使う。"""
        assert self.enc is not None, "GPU に載せた置き場からは取り出せない"
        return self.store.decode_np(self.enc[idx])


def sparse_sum(rows, W, dev=None):
    """rows: [(語の番号の並び, 重みの並び)]。行ごとに Σ 重み × W[語] の (行数 × D)。"""
    V, D = W.shape
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for r, (ids, _) in enumerate(rows):
        indptr[r + 1] = indptr[r] + len(ids)
    ids = np.fromiter((j for r in rows for j in r[0]), dtype=np.int64, count=indptr[-1])
    wts = np.fromiter((w for r in rows for w in r[1]), dtype=np.float64, count=indptr[-1])
    if dev is not None:
        out = np.zeros((len(rows), D), dtype=W.dtype)
        Wt = torch.from_numpy(np.ascontiguousarray(W)).to(dev)
        step = max(1000, int(1.5e9 // (D * W.itemsize * max(1.0, indptr[-1] / max(1, len(rows))))))
        for r0 in range(0, len(rows), step):
            r1 = min(len(rows), r0 + step)
            a, b = indptr[r0], indptr[r1]
            rl = torch.from_numpy(np.repeat(np.arange(r1 - r0), np.diff(indptr[r0:r1 + 1]))).to(dev)
            S = torch.zeros((r1 - r0, D), dtype=Wt.dtype, device=dev)
            S.index_add_(0, rl, Wt[torch.from_numpy(ids[a:b]).to(dev)] * torch.from_numpy(wts[a:b]).to(dev, Wt.real.dtype)[:, None])
            out[r0:r1] = S.cpu().numpy()
            del S, rl
        del Wt
        release(dev)
        return out
    if sp is not None:
        M = sp.csr_matrix((wts, ids, indptr), shape=(len(rows), V))
        return np.asarray(M @ W)
    out = np.zeros((len(rows), D), dtype=W.dtype)
    for r, (ii, ww) in enumerate(rows):
        if len(ii):
            out[r] = (W[list(ii)] * np.asarray(ww)[:, None]).sum(0)
    return out


# ---------------------------------------------------------------- 語の波
class Waves:
    """語の波: 乱数の位相（自分の印）と、流れから学んだ波（同じ行・隣の行の語の波の重ね合わせから共通の向きを引いた物）。
    I47a・I47b の Waves と同じ学び方。学びは complex128 で行い、しまう時に Store で持ち方を変える。"""

    def __init__(self, vocab, D=512, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab, self.D = vocab, D
        self.ix = {w: i for i, w in enumerate(vocab)}
        self.base = np.exp(1j * rng.uniform(0, 2 * np.pi, (len(vocab), D)))
        self.W = self.base.copy()
        self.spare = []

    def reserve(self, n, seed=1):
        """昼に出会う知らない語のための空き。乱数の印だけを先に作っておく（学んだ波はまだ無いので、印そのまま）。"""
        rng = np.random.default_rng(seed + len(self.vocab))
        new = np.exp(1j * rng.uniform(0, 2 * np.pi, (n, self.D)))
        self.spare = list(range(len(self.vocab) + n - 1, len(self.vocab) - 1, -1))
        self.vocab = list(self.vocab) + [None] * n
        self.base = np.concatenate([self.base, new])
        self.W = np.concatenate([self.W, new])

    def add_word(self, w):
        """知らない語に、空きの印を割り当てる。空きが無ければ None（夜の学び直しまで使えない）。"""
        if w in self.ix:
            return self.ix[w]
        if not self.spare:
            return None
        j = self.spare.pop()
        self.ix[w], self.vocab[j] = j, w
        return j

    def flow_sums(self, seq, dev=None):
        """acc[j] = Σ（語 j の出た行）（その行と前後の行の語の波の和 − 自分の波）、n[j] = その相手の数。"""
        ctx_rows, own_rows, sz = [], [], []
        for k, ws in enumerate(seq):
            ctx = set(ws) | set(seq[k - 1] if k else []) | set(seq[k + 1] if k + 1 < len(seq) else [])
            ids = [self.ix[w] for w in ctx if w in self.ix]
            if len(ids) < 2:
                continue
            ctx_rows.append(ids)
            own_rows.append([self.ix[w] for w in set(ws) if w in self.ix])
            sz.append(len(ids) - 1)
        V = len(self.vocab)
        n = np.zeros(V)
        own = np.zeros(V)
        for o, s in zip(own_rows, sz):
            n[o] += s
            own[o] += 1
        S = sparse_sum([(c, [1.0] * len(c)) for c in ctx_rows], self.W, dev)      # 行ごとの文脈の波
        # acc = Σ_行 [語 j がその行に居る] × S[行]  （S の行を、行に居る語へ足す）
        acc = np.zeros_like(self.W)
        if sp is not None and dev is None:
            indptr = np.cumsum([0] + [len(o) for o in own_rows])
            M = sp.csr_matrix((np.ones(indptr[-1]), np.fromiter((j for o in own_rows for j in o), np.int64), indptr), shape=(len(own_rows), V))
            acc = np.asarray(M.T @ S)
        elif dev is not None:
            At = torch.zeros(self.W.shape, dtype=torch.complex128, device=dev)
            step = max(1000, int(1e9 // (self.D * 16 * 4)))
            for r0 in range(0, len(own_rows), step):
                part = own_rows[r0:r0 + step]
                rl = torch.tensor([r for r, o in enumerate(part) for _ in o], dtype=torch.long, device=dev)
                wi = torch.tensor([j for o in part for j in o], dtype=torch.long, device=dev)
                St = torch.from_numpy(S[r0:r0 + step]).to(dev)
                At.index_add_(0, wi, St[rl])
                del St, rl, wi
            acc = At.cpu().numpy()
            del At
            release(dev)
        else:
            for o, s in zip(own_rows, S):
                acc[o] += s
        acc -= own[:, None] * self.W
        return acc, n

    def learn(self, seq, mix=0.6, rounds=2, dev=None):
        for r in range(rounds):
            t0 = time.time()
            acc, n = self.flow_sums(seq, dev)
            m = n > 0
            C = np.zeros_like(self.W)
            C[m] = acc[m] / n[m, None]
            C[m] -= C[m].mean(0)                            # 共通の向き（搬送波）を引く
            C[m] /= (np.linalg.norm(C[m], axis=1, keepdims=True) / math.sqrt(self.D) + 1e-9)
            W = (1 - mix) * self.base + mix * C
            self.W = W / np.maximum(np.abs(W), 1e-9)
            log.info("語の波を学んだ: %d 回目／%d（%.0f 秒、語 %d、D %d）", r + 1, rounds, time.time() - t0, len(self.vocab), self.D)


# ---------------------------------------------------------------- そろえる
def even_counter(c):
    """I47b の表の板のそろえ方: 強い所（平均より上）の平均を 1 にし、0 より 1 に近い所（> 0.5）だけ残す。"""
    if not c:
        return c
    vals = np.array(list(c.values()))
    strong = vals[vals >= vals.mean()]
    s = strong.mean() if len(strong) else 1.0
    return Counter({w: v / s for w, v in c.items() if abs(1.0 - v / s) < abs(0.0 - v / s)})


def even_dense(v):
    """読み出した明るさ（語ごと、0 以上）で同じそろえ方。明るさ 0 の所は数えない（表の板に居ない語と同じ扱い）。"""
    nz = v > 0
    if not nz.any():
        return v
    vals = v[nz]
    strong = vals[vals >= vals.mean()]
    s = strong.mean() if len(strong) else 1.0
    out = v / s
    out[out <= 0.5] = 0.0
    return out


def even_freq(S):
    """周波数ごとにそろえる: 枠ごとの大きさ |S_f| で、強い枠の平均を 1 にし、0.5 以下の枠を消す（位相はそのまま）。語に戻さない。"""
    m = np.abs(S)
    out = np.zeros_like(S)
    for i in range(len(S)):
        mi = m[i]
        strong = mi[mi >= mi.mean()]
        s = strong.mean() if len(strong) else 1.0
        keep = mi / s > 0.5
        out[i, keep] = S[i, keep] / s
    return out


# ---------------------------------------------------------------- 板
class WaveBoard:
    """会話の資料から作る、波のまま持つ板。

    fit(works)            : 語の波を流れから学び、場面（台詞 → 次の台詞）を層ごとの窓の波にしてしまう。
    add_layer(win)        : 窓 win 行の層を足す（しまってある行から作る）。
    scoop(ctxs)           : 層ごとの問いの波 → 層ごとのすくった波 S_l（次の行の波を共鳴の強さの 2 乗で重ねた物）と、選んだ場面。
    combine(...)          : 層の波を 1 本にする（重ねる／読み出してそろえる／周波数ごとにそろえる）。波のまま返す。
    read(F, k)            : 波を語の波と共鳴させて、明るい順に k 語（語に戻すのは、ここだけ）。
    """

    def __init__(self, D=512, store="c64", wins=WINS, devices=(None,), top=30, mix=0.6, seed=0, lam=0.5, keep_dir=True, spare=0):
        self.D, self.wins, self.devices, self.top, self.mix, self.seed, self.lam = D, list(wins), list(devices), top, mix, seed, lam
        self.wstore = Store(store, unit=True)
        self.kstore = Store(store)
        self.banks = []
        self._locks = {}
        self.keep_dir = keep_dir
        self.spare = spare                                  # 昼に足す語のための空き（0 なら昼の足しを使わない）
        self.day_K, self.day_B, self.day_Bd, self.day_nB = [], [], [], []
        self.day_works, self.day_from = {}, None

    # ---- 学び ----
    def fit(self, works, waves=None):
        t0 = time.time()
        self.works = works
        self.cw = [[content(ln["toks"]) for ln in wk["lines"]] for wk in works]
        seq = [ws for c in self.cw for ws in c]
        df = Counter(w for ws in seq for w in set(ws))
        self.idf = {w: math.log((len(seq) + 1) / (c + 0.5)) for w, c in df.items()}
        if waves is None:
            waves = Waves(sorted(df), D=self.D, seed=self.seed)
            waves.learn(seq, mix=self.mix, dev=self.devices[0])
        self.waves = waves
        self.scenes = []                                    # (作品, 行): 行 i の台詞が、行 i−1 の台詞を受けている
        for wi, wk in enumerate(works):
            L = wk["lines"]
            c = self.cw[wi]
            for i in range(1, len(L)):
                if L[i - 1]["text"].startswith("「") and L[i]["text"].startswith("「") and c[i - 1] and c[i]:
                    self.scenes.append((wi, i))
        nxt = [sorted(set(self.cw[wi][i])) for wi, i in self.scenes]
        freq = Counter(w for wi, i in self.scenes for w in self.cw[wi][i])   # I47b と同じ数え方（行の中の繰り返しも数える）
        self.answer_vocab = sorted(freq)                    # 読み出す語（次の行に出た語）
        self.aix = {w: j for j, w in enumerate(self.answer_vocab)}
        aid = [waves.ix[w] for w in self.answer_vocab]
        if self.spare:                                      # 昼の空き: 語の印と、読み出す語の席
            if not waves.spare:
                waves.reserve(self.spare)
            self.a_spare = list(range(len(aid) + self.spare - 1, len(aid) - 1, -1))
            self.answer_vocab = self.answer_vocab + [None] * self.spare
            self.idf_new = math.log((len(seq) + 1) / 0.5)   # 初めての語の珍しさ（1 度も見ていない語と同じ）
        self.readers = {}
        self._A = waves.base[aid].astype(np.complex64)      # 読み出しは自分の印（乱数の位相。ほかの語と重ならないので、混信はただの雑音になる）
        self._Aw = waves.W[aid].astype(np.complex64)        # 向きを測る時は学んだ波（似た語どうしが重なる）
        if self.spare:                                      # 空きの席には、どの語とも違う乱数の印を入れておく（語が決まったら、その語の印に替える）
            fill = np.exp(1j * np.random.default_rng(self.seed + 7).uniform(0, 2 * np.pi, (self.spare, self.D))).astype(np.complex64)
            self._A, self._Aw = np.concatenate([self._A, fill]), np.concatenate([self._Aw, fill])
        self.prior = [w for w, _ in freq.most_common(200)]
        self.common = set(self.prior[:50])
        z = sum(freq.values())
        self.pf = np.zeros(len(self.answer_vocab))
        self.n_main = len(self.scenes)
        for w in self.prior:
            self.pf[self.aix[w]] = freq[w] / z
        self.pf_c = Counter({w: freq[w] / z for w in self.prior})
        # 次の行の波: 次の行の語の波の和（重み 1。表の板の「語ごとに 1 回」と同じ）。共鳴の鍵（場面の波）は学んだ波、
        # すくう中身（次の行の波）は自分の印で持つ。向きを測る分だけ、学んだ波の次の行も持つ（--no_dir で省ける）。
        rows = [([waves.ix[w] for w in b], [1.0] * len(b)) for b in nxt]
        Bw = sparse_sum(rows, waves.base.astype(self.kstore.cdtype), self.devices[0])
        self.B = Bank(Bw, self.kstore, None, name="次の行")             # 次の行の波は PC のメモリ（選んだ場面の分だけ取り出す）
        self.Bd = None
        if self.keep_dir:
            Bw = sparse_sum(rows, waves.W.astype(self.kstore.cdtype), self.devices[0])
            self.Bd = Bank(Bw, self.kstore, None, name="次の行（学んだ波）")
        self.nB = np.array([len(b) for b in nxt], dtype=np.float64)
        self.nxt = nxt
        del Bw
        for w in list(self.wins):
            self._build_layer(w)
        log.info("板を作った: 場面 %d、語 %d（読み出す語 %d）、層 %s、%.0f 秒。%s", len(self.scenes), len(waves.vocab),
                 len(self.answer_vocab), self.wins, time.time() - t0, gpu_note(self.devices))
        return self

    def _build_layer(self, win):
        t0 = time.time()
        li = len(self.banks)
        dev = self.devices[li % len(self.devices)]
        rows = []
        for wi, i in self.scenes[: self.n_main]:            # 本体の場面（昼に足した分は短期の置き場に別に持つ）
            ws = sum(self.cw[wi][max(0, i - win):i], [])
            rows.append(self.bag_ids(ws))
        K = sparse_sum(rows, self.waves.W.astype(self.kstore.cdtype), dev)
        K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-9
        self.banks.append(Bank(K, self.kstore, dev, name=f"層 {win} 行"))
        self.day_K.append([q.astype(np.complex64) if (q := self.query(sum(self.cw[wi][max(0, i - win):i], []))) is not None
                           else np.zeros(self.D, np.complex64) for wi, i in self.scenes[self.n_main:]])
        del K
        if dev not in self.readers:
            self.readers[dev] = Reader(self._A, self.wstore, dev)
        log.info("層 %d 行を作った（%s、%.1f GB、%.0f 秒）", win, dev or "PC", self.banks[-1].nbytes / 2**30, time.time() - t0)

    def add_layer(self, win):
        """層を後から足す。今の場面と語の波はそのままで、窓 win 行の場面の波を作ってしまう。"""
        self.wins.append(win)
        self._build_layer(win)

    # ---- 保存と読み込み ----
    def save(self, path):
        """作った板（本体）を保存する。場面の波はしまった形（--store の持ち方）のまま、語の波は complex64 で .npz に、語や場面の表は .json に書く。
        昼に足した分は入れない（save_day の JSONL に書き、夜の学び直しで本体に入る）。"""
        n = self.n_main
        arr = dict(base=self.waves.base.astype(np.complex64), W=self.waves.W.astype(np.complex64), A=self._A, Aw=self._Aw,
                   B=self.B.dump(), nB=self.nB, pf=self.pf)
        if self.Bd is not None:
            arr["Bd"] = self.Bd.dump()
        for l, b in enumerate(self.banks):
            arr[f"K{l}"] = b.dump()
        np.savez(path + ".npz", **arr)
        meta = dict(D=self.D, store=self.wstore.name, wins=self.wins, top=self.top, mix=self.mix, seed=self.seed, lam=self.lam,
                    spare=self.spare, vocab=self.waves.vocab, w_spare=self.waves.spare, answer_vocab=self.answer_vocab,
                    a_spare=getattr(self, "a_spare", []), idf_new=getattr(self, "idf_new", 0.0), scenes=self.scenes[:n],
                    idf=self.idf, prior=self.prior, pf_c=dict(self.pf_c), n_works=self.day_from if self.day_from is not None else len(self.works))
        json.dump(meta, open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path, works, devices=(None,)):
        """保存した板を読み込む。works は作った時と同じ資料（行の文と語の解析は資料から取る）。語の波の学びと層づくりをしない。"""
        t0 = time.time()
        m = json.load(open(path + ".json", encoding="utf-8"))
        z = np.load(path + ".npz")
        assert len(works) == m["n_works"], "板を作った時と資料の数が違う"
        bd = cls(D=m["D"], store=m["store"], wins=m["wins"], devices=devices, top=m["top"], mix=m["mix"], seed=m["seed"], lam=m["lam"],
                 keep_dir="Bd" in z.files, spare=m["spare"])
        wv = Waves.__new__(Waves)
        wv.vocab, wv.D, wv.spare = m["vocab"], m["D"], m["w_spare"]
        wv.ix = {w: i for i, w in enumerate(wv.vocab) if w is not None}
        # 語の波は complex64 で保存する（位相 8 ビットで書くと、読み込んだ後の問いがずれる）。前の形も読める
        wv.base, wv.W = [x if np.iscomplexobj(x) else bd.wstore.decode_np(x) for x in (z["base"], z["W"])]
        bd.waves, bd.works = wv, list(works)
        bd.cw = [[content(ln["toks"]) for ln in wk["lines"]] for wk in works]
        bd.idf, bd.idf_new, bd.prior = m["idf"], m["idf_new"], m["prior"]
        bd.scenes = [tuple(x) for x in m["scenes"]]
        bd.n_main = len(bd.scenes)
        bd.nxt = [sorted(set(bd.cw[wi][i])) for wi, i in bd.scenes]
        bd.answer_vocab, bd.a_spare = m["answer_vocab"], m["a_spare"]
        bd.aix = {w: j for j, w in enumerate(bd.answer_vocab) if w is not None}
        bd._A, bd._Aw, bd.nB, bd.pf = z["A"], z["Aw"], z["nB"], z["pf"]
        bd.common, bd.pf_c = set(bd.prior[:50]), Counter(m["pf_c"])
        bd.B = Bank(None, bd.kstore, None, name="次の行", enc=z["B"])
        bd.Bd = Bank(None, bd.kstore, None, name="次の行（学んだ波）", enc=z["Bd"]) if "Bd" in z.files else None
        bd.readers = {}
        for l, win in enumerate(bd.wins):
            dev = bd.devices[l % len(bd.devices)]
            bd.banks.append(Bank(None, bd.kstore, dev, name=f"層 {win} 行", enc=z[f"K{l}"]))
            bd.day_K.append([])
            if dev not in bd.readers:
                bd.readers[dev] = Reader(bd._A, bd.wstore, dev)
        log.info("板を読み込んだ: 場面 %d、層 %s、%.0f 秒。%s", bd.n_main, bd.wins, time.time() - t0, gpu_note(bd.devices))
        return bd

    def bag_ids(self, ws):
        ids = [self.waves.ix[w] for w in ws if w in self.waves.ix]
        return ids, [self.idf.get(w, 1.0) for w in ws if w in self.waves.ix]

    def query(self, ws):
        ids, wt = self.bag_ids(ws)
        if not ids:
            return None
        v = (self.waves.W[ids] * np.asarray(wt)[:, None]).sum(0)
        return v / (np.linalg.norm(v) + 1e-9)

    # ---- すくう ----
    def scoop_layer(self, l, Q):
        """層 l で、問いの波 Q（問い × D）ごとに共鳴の強い場面を選び、次の行の波を強さの 2 乗で重ねる。"""
        dev = self.devices[l % len(self.devices)]
        with self._locks.setdefault(dev, threading.Lock()):   # 同じ GPU の層は 1 つずつすくう（同時だと作業用のメモリが足りなくなる）
            idx, r = self.banks[l].top(Q, self.top)
        if self.day_B:                                      # 昼に足した場面（短期の置き場）も同じ問いで共鳴させ、強い順に合わせる
            Kd = np.stack(self.day_K[l])
            rd = np.abs(Kd @ np.conj(Q).T.astype(np.complex64)).T
            V = np.concatenate([r, rd], 1)
            I = np.concatenate([idx, np.broadcast_to(self.n_main + np.arange(len(Kd)), rd.shape)], 1)
            o = np.argsort(-V, 1)[:, : self.top]
            idx, r = np.take_along_axis(I, o, 1), np.take_along_axis(V, o, 1)
        w2 = r ** 2
        S = np.einsum("qk,qkd->qd", w2.astype(np.float32), self.brows(idx))
        Sd = np.einsum("qk,qkd->qd", w2.astype(np.float32), self.brows(idx, True)) if self.Bd is not None else None
        Z = (w2 * self.nb(idx)).sum(1)                      # 表の板で言えば、明るさの合計
        return S, Z, idx, w2, Sd

    def brows(self, idx, learned=False):
        """場面の番号 → 次の行の波（本体と、昼に足した分）。"""
        bank, day = (self.Bd, self.day_Bd) if learned else (self.B, self.day_B)
        idx = np.asarray(idx)
        if not day:
            return bank.rows(idx)
        m = idx >= self.n_main
        out = np.zeros(idx.shape + (self.D,), np.complex64)
        out[~m] = bank.rows(idx[~m])
        if m.any():
            out[m] = np.stack(day)[idx[m] - self.n_main]
        return out

    def nb(self, idx):
        return (np.concatenate([self.nB, self.day_nB]) if self.day_nB else self.nB)[idx]

    # ---- 昼: 場面を 1 つずつ足す ----
    def add_line(self, conv, text, toks):
        """会話 conv（名前）に 1 行足す。前の行と今の行に中身の語があれば、新しい場面として短期の置き場に入れる。
        今の語の波のまま足すだけで、学び直さない。知らない語には空きの印を割り当てる。足した場面はすぐにすくえる。
        返り値: 場面を足したか。"""
        assert self.spare, "昼の足しを使うには WaveBoard(spare=…) で空きを作る"
        if conv not in self.day_works:
            self.day_works[conv] = len(self.works)
            if self.day_from is None:
                self.day_from = len(self.works)
            self.works.append(dict(name=conv, lines=[]))
            self.cw.append([])
        wi = self.day_works[conv]
        ws = content(toks)
        for w in ws:
            if self.waves.add_word(w) is not None:
                self.idf.setdefault(w, self.idf_new)
        self.works[wi]["lines"].append(dict(text=text if text.startswith("「") else "「" + text + "」", toks=toks))
        self.cw[wi].append(ws)
        c = self.cw[wi]
        i = len(c) - 1
        b = sorted({w for w in ws if w in self.waves.ix})
        if i < 1 or not c[i - 1] or not b:
            return False
        keys = [self.query(sum(c[max(0, i - w):i], [])) for w in self.wins]
        if keys[0] is None:
            return False
        for w in b:                                         # 返事に出た語を、読み出せる語にする
            if w not in self.aix and self.a_spare:
                j = self.a_spare.pop()
                self.aix[w], self.answer_vocab[j] = j, w
                self._A[j], self._Aw[j] = self.waves.base[self.waves.ix[w]], self.waves.W[self.waves.ix[w]]
                for r in self.readers.values():
                    r.set_row(j, self._A[j])
        b = [w for w in b if w in self.aix]
        if not b:
            return False
        ids = [self.waves.ix[w] for w in b]
        for l, k in enumerate(keys):
            self.day_K[l].append(k.astype(np.complex64))
        self.day_B.append(self.waves.base[ids].sum(0).astype(np.complex64))
        if self.Bd is not None:
            self.day_Bd.append(self.waves.W[ids].sum(0).astype(np.complex64))
        self.day_nB.append(float(len(b)))
        self.nxt.append(b)
        self.scenes.append((wi, i))
        return True

    def save_day(self, path):
        """昼に足した会話を、資料と同じ形（作品 = 会話、行 = 文と語の解析）で書く。夜の学び直しで --day に渡す。"""
        with open(path, "w", encoding="utf-8") as f:
            for wi in self.day_works.values():
                f.write(json.dumps(self.works[wi], ensure_ascii=False) + "\n")

    def scoop(self, ctx_q):
        """ctx_q: 層ごとの問いの波（問い × D、無い問いは 0）。層ごとに別の GPU で同時に回す。"""
        with ThreadPoolExecutor(max_workers=len(self.banks)) as ex:
            return list(ex.map(lambda l: self.scoop_layer(l, ctx_q[l]), range(len(self.banks))))

    def reader(self, l=0):
        return self.readers[self.banks[l].dev] if self.banks else next(iter(self.readers.values()))

    def read_dense(self, S, l=0, floor=True):
        """波 → 語ごとの明るさ（問い × 読み出す語）。Re<語の印, S>/D。
        雑音の床: 波に居ない語の読みは、平均 0・ばらつき σ = |S|/(D√2) の雑音になる（語の印が乱数の位相だから）。
        読み出す語が V 個ある時、雑音だけで出るいちばん大きい読みの見込みは σ√(2 ln V)。それより下は消す。
        手で置く値ではなく、波の強さ・長さ・語の数から決まる。D が大きいほど床が下がり、弱い語まで読める。"""
        v = self.reader(l).scores(S)
        if not floor:
            return np.maximum(v, 0.0)
        fl = np.linalg.norm(S, axis=1) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(self.answer_vocab)))
        v[v < fl[:, None]] = 0.0
        return v

    def resuperpose(self, E, learned=False):
        """語ごとの明るさ E（問い × 読み出す語）→ 波（Σ 明るさ × 語の波）。読み出してそろえた後に波へ戻す。
        learned=True は学んだ波で重ねる（向きを測る時）。"""
        rows = [(np.flatnonzero(e), e[np.flatnonzero(e)]) for e in E]
        return sparse_sum(rows, self._Aw if learned else self._A, None)

    def rank(self, v, k=10):
        """読み出した明るさと、ふだんの多さを混ぜて上位 k 語（I47b の rank と同じ）。"""
        s = v.sum()
        sc = (1 - self.lam) * (v / s if s > 0 else v) + self.lam * self.pf
        top = np.argpartition(-sc, k)[:k]
        return [self.answer_vocab[j] for j in top[np.argsort(-sc[top])] if self.answer_vocab[j] is not None]

    def rank_counter(self, c, k=10):
        s = sum(c.values()) or 1.0
        sc = Counter({w: (1 - self.lam) * v / s for w, v in c.items()})
        for w, p in self.pf_c.items():
            sc[w] += self.lam * p
        return [w for w, _ in sc.most_common(k)]

    def free(self):
        for b in self.banks:
            b.free()
        for r in self.readers.values():
            r.free()


class Reader:
    """語の波（読み出す語 × D）を GPU に置き、すくった波を語の明るさにする（語に戻す所）。"""

    def __init__(self, A, store, dev):
        self.store, self.dev = store, dev
        self.enc = store.encode(A)
        self.D = A.shape[1]
        self.t = None
        if dev is not None:
            self.t = torch.from_numpy(self.enc).to(dev)

    def scores(self, S):
        if self.t is None:
            A = self.store.decode_np(self.enc)
            return (np.conj(A) @ S.T).real.T / self.D
        cdt = torch.complex128 if self.store.name == "c128" else torch.complex64
        A = self.store.decode_t(self.t).to(cdt)
        St = torch.from_numpy(np.ascontiguousarray(S)).to(self.dev, cdt)
        out = (St @ A.conj().T).real / self.D
        return out.cpu().numpy().astype(np.float64)

    def set_row(self, j, a):
        """読み出す語の席 j に、語の印を入れる（昼に足した語）。"""
        e = self.store.encode(a[None])
        self.enc[j] = e[0]
        if self.t is not None:
            self.t[j] = torch.from_numpy(e).to(self.dev)[0]

    def free(self):
        self.t = None
        release(self.dev)


# ---------------------------------------------------------------- 返事作りにつなぐ
class WaveSource:
    """bot の返事作りにつなぐ口。会話のここまでの行（中身の語の並び、古い順）から、返事の中身の元になる行を出す。

    1. 層ごとの窓（前の 1・3・10・30 行…）の波で場面をすくう（すくった物は波のまま）。
    2. 層ごとに読み出してそろえ、重ねてもう一度そろえ、波 F に重ね直す（返事の意味の波）。
    3. すくった場面の「次の行」の波と F を共鳴させ、強い順に n 行を返す（重み = 共鳴の強さ、いちばん強い行が 1）。
       行を選ぶ所まで波で行い、語に戻した物（words）は記録に出すだけ。
    返す行は (重み, 文, 語の解析)。chat_bot3 の extra_lines から場の板に入る（行づくりの中身の元）。"""

    def __init__(self, board, n=12):
        self.board, self.n = board, n
        self.note = ""

    def lines(self, ctx):
        bd = self.board
        L = len(bd.banks)
        Z = np.zeros(bd.D, dtype=np.complex128)
        qs = [bd.query(sum(ctx[-w:], [])) for w in bd.wins]
        if qs[0] is None:
            self.note = ""
            return []
        res = bd.scoop([(q if q is not None else Z)[None] for q in qs])
        E = np.zeros((1, len(bd.answer_vocab)))
        for l in range(L):
            if qs[l] is not None:
                E += even_dense(bd.read_dense(res[l][0] / max(res[l][1][0], 1e-12), l)[0])[None]
        E[0] = even_dense(E[0])
        F = bd.resuperpose(E)[0]                            # 返事の意味の波
        cand = np.unique(np.concatenate([res[l][2][0] for l in range(L) if qs[l] is not None]))
        sc = (bd.brows(cand) @ np.conj(F)).real / (bd.D * np.sqrt(np.maximum(bd.nb(cand), 1.0)))
        top = [(w, float(E[0, j])) for j, w in ((j, bd.answer_vocab[j]) for j in np.argsort(-E[0])[:8]) if E[0, j] > 0 and w is not None]
        out, seen = [], set()
        mx = max(float(sc.max()), 1e-12)
        for o in np.argsort(-sc):
            if sc[o] <= 0 or len(out) >= self.n:
                break
            wi, i = bd.scenes[cand[o]]
            ln = bd.works[wi]["lines"][i]
            if ln["text"] in seen:
                continue
            seen.add(ln["text"])
            out.append((float(sc[o]) / mx, ln["text"], ln["toks"]))
        self.note = "意味の波ですくった語「" + "・".join(w for w, _ in top) + f"」、行 {len(out)}"
        return out


# ---------------------------------------------------------------- 測る
def split(works, a):
    works = list(works)
    random.Random(a.seed).shuffle(works)
    tr_w, te_w = works[a.test_works:], works[:a.test_works]
    if a.train_works:
        tr_w = tr_w[: a.train_works]
    return tr_w, te_w


def test_items(board, te_w, wins):
    out = []
    for wk in te_w:
        L = wk["lines"]
        cw = [content(ln["toks"]) for ln in L]
        for i in range(1, len(L)):
            if L[i - 1]["text"].startswith("「") and L[i]["text"].startswith("「") and cw[i - 1] and cw[i]:
                out.append(([sum(cw[max(0, i - w):i], []) for w in wins], cw[i]))
    return out


MODES = ("1 層（今の行だけ）", "4 層を重ねる（そろえない）", "4 層、すくうたびにそろえる", "今の行 + 場面（1・10 行）をそろえる")


def evaluate(board, test, a, rng_seed=1):
    """表の板（I47b と同じ）と波の板を、同じ共鳴の結果から測る。"""
    t0 = time.time()
    L = len(board.banks)
    i10 = board.wins.index(10) if 10 in board.wins else min(2, L - 1)
    Z = np.zeros(board.D, dtype=np.complex128)
    Qs = [np.stack([q if (q := board.query(ctx[l])) is not None else Z for ctx, _ in test]) for l in range(L)]
    TB = np.stack([q if (q := board.query(b)) is not None else Z for _, b in test]).astype(np.complex64)   # 向きの候補（本当の次の行の波）
    rr = np.random.default_rng(rng_seed)
    cand = [np.array([i] + [j for j in rr.choice(len(test), 120, replace=False) if j != i][:99]) for i in range(len(test))]
    kinds = ["表", "波・読み出してそろえる", "波・周波数ごとにそろえる"]
    tot = {(k, m): np.zeros(6) for k in kinds for m in MODES}   # 再現率, 当たり, 中身, 中身の数, 板の語, 向き
    agree = {(k, m): [0.0, 0] for k in kinds[1:] for m in MODES}
    tot[("表", "ふだんの多さ")] = np.zeros(6)
    bs = a.batch
    for b0 in range(0, len(test), bs):
        b1 = min(len(test), b0 + bs)
        res = board.scoop([Qs[l][b0:b1] for l in range(L)])
        S = [r[0] / np.maximum(r[1], 1e-12)[:, None] for r in res]       # 層ごとの波（明るさの合計で割る = 表の板の割合と同じ）
        Sd = [r[4] / np.maximum(r[1], 1e-12)[:, None] for r in res] if board.Bd is not None else None
        ev = lambda E: np.stack([even_dense(x) for x in E])
        K1, K2 = kinds[1], kinds[2]
        # 波の形。語に戻す（読み出す）のは、重ねる形では最後の 1 回、読み出してそろえる形では層ごとと最後。
        R = [board.read_dense(S[l], l) for l in range(L)]
        E4 = [ev(Rl) for Rl in R]
        Sf = [even_freq(x) for x in S]
        F = {(K1, MODES[0]): S[0], (K1, MODES[1]): sum(S),
             (K1, MODES[2]): board.resuperpose(sum(E4)), (K1, MODES[3]): board.resuperpose(E4[0] + E4[i10]),
             (K2, MODES[0]): S[0], (K2, MODES[1]): sum(S),
             (K2, MODES[2]): sum(Sf), (K2, MODES[3]): Sf[0] + Sf[i10]}
        dense = {}
        for key, Fw in F.items():
            if key[1] == MODES[0]:
                dense[key] = R[0]
            else:
                dense[key] = board.read_dense(Fw)            # 最後の読み出し
                if key[1] in MODES[2:]:
                    dense[key] = ev(dense[key])              # 重ねた後のそろえ
        Fd = {}
        if Sd is not None:                                   # 向きを測る波（学んだ波の側で同じ形を作る）
            Sdf = [even_freq(x) for x in Sd]
            Fd = {(K1, MODES[0]): Sd[0], (K1, MODES[1]): sum(Sd), (K2, MODES[0]): Sd[0], (K2, MODES[1]): sum(Sd),
                  (K1, MODES[2]): board.resuperpose(dense[(K1, MODES[2])], True), (K1, MODES[3]): board.resuperpose(dense[(K1, MODES[3])], True),
                  (K2, MODES[2]): even_freq(sum(Sdf)), (K2, MODES[3]): even_freq(Sdf[0] + Sdf[i10])}
        tabsB, tabE = [], {m: np.zeros((b1 - b0, len(board.answer_vocab)), np.float32) for m in MODES}
        for qi in range(b1 - b0):                            # 表の板（I47b と同じ計算）
            cs = []
            for l in range(L):
                c = Counter()
                for j, rj in zip(res[l][2][qi], res[l][3][qi]):
                    for w in board.nxt[j]:
                        c[w] += rj
                cs.append(c)
            tabs = {MODES[0]: cs[0],
                    MODES[1]: sum((Counter({w: v / (sum(c.values()) or 1) for w, v in c.items()}) for c in cs), Counter()),
                    MODES[2]: even_counter(sum((even_counter(c) for c in cs), Counter())),
                    MODES[3]: even_counter(even_counter(cs[0]) + even_counter(cs[i10]))}
            tabsB.append(tabs)
            for m, c in tabs.items():
                for w, v in c.items():
                    tabE[m][qi, board.aix[w]] = v
        Ft = {m: board.resuperpose(tabE[m], True) for m in MODES} if Sd is not None else {}
        for qi in range(b1 - b0):
            ti = b0 + qi
            b = test[ti][1]
            B, Bi = set(b), set(b) - board.common

            def score(key, top, nboard, Fv):
                t = tot[key]
                t[0] += len(B & top) / len(B)
                t[1] += bool(B & top)
                if Bi:
                    t[2] += len(Bi & top) / len(Bi)
                    t[3] += 1
                t[4] += nboard
                if Fv is not None:
                    s_ = np.abs(TB[cand[ti]] @ np.conj(Fv))
                    t[5] += (s_[1:] >= s_[0]).sum() < 10
            score(("表", "ふだんの多さ"), set(board.prior[:10]), 0, None)
            for m, c in tabsB[qi].items():
                ttop = board.rank_counter(c)
                score(("表", m), set(ttop), len(c), Ft[m][qi] if Ft else None)
                for k in kinds[1:]:
                    v = dense[(k, m)][qi]
                    wtop = board.rank(v)
                    score((k, m), set(wtop), int((v > 0).sum()), Fd[(k, m)][qi] if Fd else None)
                    agree[(k, m)][0] += len(set(wtop) & set(ttop))
                    agree[(k, m)][1] += 1
        done = b1
        el = time.time() - t0
        log.info("測り: %d/%d 問（%.0f 秒、残り %.0f 秒ほど）。%s", done, len(test), el, el / done * (len(test) - done), gpu_note(board.devices))
    n = len(test)
    out = {}
    for (k, m), t in tot.items():
        out[f"{k}／{m}"] = dict(再現率=round(t[0] / n, 4), 当たり=round(t[1] / n, 4), 中身=round(t[2] / max(1, t[3]), 4),
                               板の語=round(t[4] / n, 1), 向き=round(t[5] / n, 4))
    out["表と波の上位 10 語の重なり（10 が全部同じ）"] = {f"{k}／{m}": round(v[0] / max(1, v[1]), 2) for (k, m), v in agree.items()}
    out["測りの秒"] = round(time.time() - t0, 1)
    return out


NIGHT_KEYS = (("波・周波数ごとにそろえる／" + MODES[2], "当たり"), ("波・読み出してそろえる／" + MODES[1], "向き"))


def night_verdict(prev, now):
    """前の日の板と今日学び直した板を、同じ試しで比べる。当たりか向きが「揺れより大きく」下がっていたら戻す。
    揺れ = 同じ試しの組で 2 つの割合を比べる時のばらつき √(p(1−p)·2/n) の 2 倍（手で置く値ではなく、試しの数から決まる）。"""
    out = dict(判定="合流")
    n = min(prev["試しの組"], now["試しの組"])
    for k, m in NIGHT_KEYS:
        p0, p1 = float(prev[k][m]), float(now[k][m])
        se = 2 * math.sqrt(max(p0 * (1 - p0), 1e-9) * 2 / n)
        out[f"{k}・{m}"] = dict(前=p0, 今=p1, 揺れ=round(se, 4))
        if p1 < p0 - se:
            out["判定"] = "戻す"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--test_works", type=int, default=10)
    ap.add_argument("--train_works", type=int, default=0)
    ap.add_argument("--test_limit", type=int, default=0, help="試しの組の数を減らす（0 = 全部）")
    ap.add_argument("--D", default="512", help="波の長さ（カンマで並べると表）")
    ap.add_argument("--store", default="c64", help="数の持ち方 c128,c64,p8（カンマで並べると表）")
    ap.add_argument("--wins", default="1,3,10,30", help="層の窓（前の何行を見るか）")
    ap.add_argument("--devices", default="auto", help="auto / numpy / cuda:0,cuda:1（層を順に割り当てる）")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--mix", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256, help="まとめて共鳴させる問いの数")
    ap.add_argument("--no_dir", action="store_true", help="向きを測らない（学んだ波の次の行を持たないので、PC のメモリが半分で済む）")
    ap.add_argument("--day", help="夜の学び直し: 昼に足した会話（save_day の JSONL、カンマで並べる）を学ぶ資料に足す")
    ap.add_argument("--prev", help="夜の学び直し: 前の日の結果（--out の JSON）。当たりと向きが下がっていたら「戻す」と判定する")
    ap.add_argument("--out")
    ap.add_argument("--log")
    a = ap.parse_args()
    hs = [logging.StreamHandler()] + ([logging.FileHandler(a.log, encoding="utf-8")] if a.log else [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", handlers=hs)
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    tr_w, te_w = split(works, a)
    for path in (a.day.split(",") if a.day else []):        # 試しは固定のまま、昼の会話を学ぶ側にだけ足す
        day = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        tr_w = tr_w + day
        log.info("昼の会話を足した: %s（%d 会話）", path, len(day))
    devs = parse_devices(a.devices)
    wins = [int(x) for x in a.wins.split(",")]
    log.info("始め: 学ぶ作品 %d、試し %d、置き場 %s、torch %s", len(tr_w), len(te_w), devs, getattr(torch, "__version__", "なし"))
    table = dict(学ぶ作品=len(tr_w), 試しの作品=len(te_w), 層の窓=wins, 置き場=[str(d) for d in devs], 結果=[])
    for D in [int(x) for x in a.D.split(",")]:
        waves = None
        for st in a.store.split(","):
            t0 = time.time()
            board = WaveBoard(D=D, store=st, wins=wins, devices=devs, top=a.top, mix=a.mix, seed=a.seed, lam=a.lam, keep_dir=not a.no_dir)
            board.fit(tr_w, waves=waves)
            waves = board.waves                             # 同じ D の語の波は 1 回だけ学ぶ（持ち方だけを変える）
            test = test_items(board, te_w, wins)
            if a.test_limit:
                test = test[: a.test_limit]
            r = evaluate(board, test, a)
            r.update(D=D, 持ち方=st, 試しの組=len(test), 場面=len(board.scenes),
                     場面1つ1層のバイト=board.kstore.bytes_per(D), 語1つのバイト=board.wstore.bytes_per(D),
                     層の波の合計GB=round(sum(b.nbytes for b in board.banks) / 2**30, 3),
                     次の行の波GB=round(board.B.nbytes / 2**30, 3), 秒=round(time.time() - t0, 1))
            table["結果"].append(r)
            log.info("D %d・%s: 表 4 層そろえる 当たり %.4f、波（読み出し）%.4f、波（周波数）%.4f、向き 表（4 層そろえる）%.4f 波（1 層）%.4f",
                     D, st, r["表／" + MODES[2]]["当たり"], r["波・読み出してそろえる／" + MODES[2]]["当たり"],
                     r["波・周波数ごとにそろえる／" + MODES[2]]["当たり"], r["表／" + MODES[2]]["向き"],
                     r["波・読み出してそろえる／" + MODES[0]]["向き"])
            board.free()
            del board
            if a.out:
                json.dump(table, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.prev:                                              # 夜の判定: 前の日より下がったら、今日の分を合流させない
        table["夜の判定"] = night_verdict(json.load(open(a.prev, encoding="utf-8"))["結果"][0], table["結果"][0])
        log.info("夜の判定: %s", table["夜の判定"])
        if a.out:
            json.dump(table, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(table, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
