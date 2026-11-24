"""I49zi: 小さい自作の世界で、語の位相を誤差で動かす形を確かめる（持ち主の案の第 2 段、未確定の試し。乱数なし）。

持ち主の言葉:「位相板の周波数と初期位相を微分可能な光の物理モデルとして勾配降下法で最適化する」「段階分けで進めて」
今までの決まり（数えて足すだけ、値を直接動かさない）から外れる試し。良し悪しの判断ではなく、動かすと何が起きるかを見るための物。
世界は tools/i49q_small_world.py の world（話が輪になって語を分け合う。3 つ組の 3 分の 1 は聞かせない）。
答え方は I49k の解く形と同じ: 語の覚えた波 = 一緒に座った語の元の波を、席の組の回し方で回して足した物。
  問い = 見せた 2 語の元の波を回して足した「狙い」と、候補の覚えた波との近さ（cos の実部）。
流れ: 聞いた 3 つ組を順に、1 語ずつ隠して問い（候補はその席の語全部）、答え合わせをしてから、その 3 つ組を聞かせる。
比べる形（直すのは外れた時だけ。正解の点を上げ、選んだ語の点を下げる向きへ、長さをそろえた 1 歩）:
  固定                         何も動かさない（今の形）
  振幅だけ                     周波数ごとの読む時の振幅（隠した席ごと）だけを直す。波は動かさない（第 1 段と同じ考え方）
  位相・覚えた波はそのまま       語の元の波の位相を直す。前に足し込んだ覚えた波は、古い位相のまま残る
  位相・覚えた波を作り直す       位相を直すたびに、聞いた数の表から覚えた波を作り直す。勾配は、狙いと覚えた波の両方を通す
数える物:
  流れの当たり   流れの中で、話に合う語を選べた割合（聞く前に問う）
直し方の合図（teach）:
  隠した語      選んだ語が隠した語と違えば直す（本番の流れと同じ。話に合う別の語を選んでいても、外れとして直す）
  話に合う語    選んだ語が話に合わない時だけ直す（正しいと言える答えは、外れにしない）
  覚え当て       終わりに、聞いた 3 つ組の 1 語を隠して、話に合う語を選べた割合
  広がり当て     終わりに、聞いていないがありそうな 3 つ組の 1 語を隠して、話に合う語を選べた割合
使い方: python tools/i49zi_small_grad.py 出力.json   （記録はリポジトリの外へ）
手で置く物: 世界の形、位相の本数、くり返しの回数、直す速さ、初めの位相の決め方、長さをそろえた 1 歩にする事、振幅を 0 以上に切る事。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import i49q_small_world as SW  # noqa: E402

G, R2 = SW.G, SW.R2
PAIRS = [(a, b) for a in range(3) for b in range(3) if a != b]
FORMS = ("固定", "振幅だけ", "位相・覚えた波はそのまま", "位相・覚えた波を作り直す")


class Small:
    def __init__(self, n, K, form, lr):
        assert form in FORMS
        self.n, self.K, self.form, self.lr = n, K, form, float(lr)
        i = np.arange(1, n + 1)[:, None].astype(float)
        k = np.arange(1, K + 1)[None, :].astype(float)
        self.phi = 2 * np.pi * ((i * G * k * k + i * i * R2 * k) % 1.0)          # 語ごと・周波数ごとの位相
        self.rel = {(a, b): np.exp(2j * np.pi * ((PAIRS.index((a, b)) + 1) * G * np.arange(K))) for a, b in PAIRS}
        self.cnt = {p: np.zeros((n, n)) for p in PAIRS}                           # (自分の席, 相手の席) → 語 × 相手 の聞いた数
        self.V = np.zeros((n, K), dtype=np.complex128)                            # 覚えた波
        self.amp = np.ones((3, K))                                                # 読む時の振幅（隠した席ごと）
        self.n_fix = 0

    def B(self, w=None):
        p = self.phi if w is None else self.phi[w]
        return np.exp(1j * p) / np.sqrt(self.K)

    def rebuild(self):
        B = self.B()
        self.V = sum(self.cnt[p] @ (B * self.rel[p][None, :]) for p in PAIRS)

    def hear(self, ids):
        for a in range(3):
            for b in range(3):
                if a != b:
                    self.cnt[(a, b)][ids[a], ids[b]] += 1
                    self.V[ids[a]] += self.B(ids[b]) * self.rel[(a, b)]

    def aim(self, si, ids):
        return sum(self.B(ids[b]) * self.rel[(si, b)] for b in range(3) if b != si)

    def scores(self, si, ids, cands):
        aim = self.aim(si, ids)
        a = self.amp[si]
        V = self.V[cands]
        nv = np.sqrt(np.sum(a[None, :] * np.abs(V) ** 2, axis=1))
        na = float(np.sqrt(np.sum(a * np.abs(aim) ** 2)))
        s = np.full(len(cands), -np.inf)
        ok = nv > 0
        if na > 0:
            s[ok] = (np.real(V[ok] * np.conj(aim)[None, :]) @ a) / (na * nv[ok])
        return s, aim

    def fix(self, si, ids, cands, s, aim):
        """外れた時だけ: 正解の点を上げ、選んだ語の点を下げる向きへ 1 歩。"""
        t, w = ids[si], cands[int(np.argmax(s))]
        if self.form == "固定" or w == t or not np.isfinite(s[cands.index(t)]):
            return
        na = float(np.linalg.norm(aim))
        nt, nw = float(np.linalg.norm(self.V[t])), float(np.linalg.norm(self.V[w]))
        if min(na, nt, nw) <= 0:
            return
        d = (self.V[t] / nt - self.V[w] / nw) / na                                # 狙いをこの向きへ寄せたい
        if self.form == "振幅だけ":
            g = np.real(np.conj(aim) * d)
            n = float(np.linalg.norm(g))
            if n > 0:
                self.amp[si] = np.maximum(0.0, self.amp[si] + self.lr * np.sqrt(self.K) * g / n)
                self.n_fix += 1
            return
        grad = np.zeros_like(self.phi)
        for b in range(3):                                                        # 狙いを通る分（見せた 2 語の位相）
            if b != si:
                grad[ids[b]] += np.imag(np.conj(self.rel[(si, b)] * self.B(ids[b])) * d)
        if self.form == "位相・覚えた波を作り直す":                                # 覚えた波を通る分（正解と選んだ語の相手の位相）
            B = self.B()
            for c, nc, sign in ((t, nt, 1.0), (w, nw, -1.0)):
                for p in PAIRS:
                    row = self.cnt[p][c]
                    nz = np.nonzero(row)[0]
                    if len(nz):
                        grad[nz] += sign * row[nz, None] * np.imag(aim[None, :] * np.conj(self.rel[p][None, :] * B[nz])) / (na * nc)
        n = float(np.linalg.norm(grad))
        if n > 0:
            self.phi += self.lr * np.sqrt(self.K) * grad / n
            self.n_fix += 1
            if self.form == "位相・覚えた波を作り直す":
                self.rebuild()


def run_one(T, K, form, lr, passes, teach="隠した語"):
    assert teach in ("隠した語", "話に合う語")
    heard, plaus, odd, half, ok, seats = SW.world(T)
    words = list(dict.fromkeys(w for s in seats for w in s))
    idx = {w: i for i, w in enumerate(words)}
    cand = [[idx[w] for w in s] for s in seats]
    m = Small(len(words), K, form, lr)
    hit = tot = 0
    for _ in range(passes):
        for t in heard:
            ids = [idx[w] for w in t]
            asked = []
            for si in range(3):
                s, aim = m.scores(si, ids, cand[si])
                if np.isfinite(s).any() and float(np.max(s)) > 0:
                    best = words[cand[si][int(np.argmax(s))]]
                    fit = tuple(best if j == si else t[j] for j in range(3)) in ok
                    hit += fit
                    if teach == "隠した語" or not fit:
                        asked.append((si, s, aim))
                tot += 1
            for si, s, aim in asked:
                m.fix(si, ids, cand[si], s, aim)
            m.hear(ids)

    def guess(tris):
        h = n = 0
        for t in tris:
            ids = [idx[w] for w in t]
            for si in range(3):
                s, _ = m.scores(si, ids, cand[si])
                best = words[cand[si][int(np.argmax(s))]] if np.isfinite(s).any() else None
                h += tuple(best if j == si else t[j] for j in range(3)) in ok
                n += 1
        return h / max(1, n)

    drift = float(np.mean(np.abs(np.exp(1j * m.phi) - np.exp(1j * Small(len(words), K, "固定", 0).phi))))
    return dict(話=T, 語=len(words), 本数=K, 形=form, 合図=teach, 速さ=lr, 回=passes, 流れの当たり=round(hit / max(1, tot), 3),
                覚え当て=round(guess(heard[::max(1, len(heard) // 150)]), 3), 広がり当て=round(guess(plaus[::max(1, len(plaus) // 60)]), 3),
                直した回数=m.n_fix, 位相の動き=round(drift, 3), 振幅の大=round(float(m.amp.max()), 2))


def run(Ts=(10, 40), Ks=(4, 8, 16), lrs=(.01, .03, .1), passes_list=(1, 5)):
    rows = []
    for T in Ts:
        for K in Ks:
            for p in passes_list:
                for form in FORMS:
                    for teach in (("隠した語",) if form == "固定" else ("隠した語", "話に合う語")):
                        for lr in ((0.0,) if form == "固定" else lrs):
                            rows.append(run_one(T, K, form, lr, p, teach))
                            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    rows = run()
    json.dump(dict(rows=rows, 秒=round(time.time() - t0)), open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
