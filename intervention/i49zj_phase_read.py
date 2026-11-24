"""I49zj: 語の元の波の位相を、外れた時だけ誤差の向きへ動かす（持ち主の案の第 2 段、未確定の試し）。

持ち主の言葉:「位相板の周波数と初期位相を微分可能な光の物理モデルとして勾配降下法で最適化する」「段階分けで進めて」
今までの決まり（数えて足すだけ、値を直接動かさない）から外れる試し。採用ではなく、効くかどうかを数えるための物。
小さい世界（I49zi）で一番良かった形を、本番の解く形（I49k・I49m）に入れる:
  語ごと・周波数（属性）ごとに、位相のずれ θ を持つ（初めは 0）。語の元の波 = 属性の割合 × e^{iθ}。振幅（属性の割合）は変えない。
  外れた時だけ、見せた 2 語の位相を、正解の覚えた波へ寄り、選んだ語の覚えた波から離れる向きへ、長さをそろえた 1 歩だけ動かす。
  覚えた波（SeatVec に足し込んだ分）は作り直さない。動かした後に聞く分から、新しい位相で足される。乱数は使わない。

  PhaseRead(sv, phase_lr=0.1, teach="隠した語", **振幅の引数)      AmpRead（I49zg）を継ぐ。振幅の lr は、渡さなければ 0（振幅は直さない）
    teach:
      "隠した語"    選んだ語が隠した語と違えば直す
      "聞いた組"    選んだ語が、同じ席・同じ見せた 2 語でそれまでに聞いた事の有る語なら、外れにしない（正しいと言える答えを直さない）。
                   聞いた事の有る組は、答え合わせのたびに数えて持つ（先の話の中身は使わない）。hear_seen(3 つ組) で先に聞かせた分も入れられる。
    1 歩の大きさ: 動かす成分（見せた 2 語が持つ属性）1 つあたり、ならして phase_lr ラジアン。
    spread() に、位相の動き（動かした語の数、ずれの大きさの平均、直した回数）を足す。
手で置く物: 直す速さ phase_lr、長さをそろえた 1 歩にする事、覚えた波を作り直さない事、AmpRead の物。
"""
from __future__ import annotations

import numpy as np

from i49k_seat_vec import PAIRS
from i49zg_amp_read import AmpRead


class PhaseRead(AmpRead):
    def __init__(self, sv, phase_lr=0.1, teach="隠した語", **kw):
        assert teach in ("隠した語", "聞いた組") and sv.bind == "位相"
        kw.setdefault("lr", 0.0)
        super().__init__(sv, **kw)
        self.phase_lr, self.teach = float(phase_lr), teach
        self.theta = {}                                   # 語 → 位相のずれ（周波数ごと）
        self.seen = {}                                    # (隠した席, 見せた 2 語) → その席で聞いた事の有る語
        self.n_turn = 0

    def hear_seen(self, tri):
        t = tuple(tri)
        for si in range(3):
            self.seen.setdefault((si, t[:si] + t[si + 1:]), set()).add(t[si])

    def _turn(self, si, names, s, aim, V, tri, truth):
        if self.phase_lr <= 0 or truth not in names:
            return
        t, w = names.index(truth), int(np.argmax(s))
        if w == t:
            return
        if self.teach == "聞いた組" and names[w] in self.seen.get((si, tri[:si] + tri[si + 1:]), ()):
            return
        na, nt, nw = float(np.linalg.norm(aim)), float(np.linalg.norm(V[t])), float(np.linalg.norm(V[w]))
        if min(na, nt, nw) <= 0:
            return
        d = (V[t] / nt - V[w] / nw) / na                  # 狙いをこの向きへ寄せたい
        gs, nnz, sq = [], 0, 0.0
        for b in range(3):
            if b == si:
                continue
            bp = self.sv._base(tri[b])
            if bp is None:
                continue
            self.sv.rot(bp, (si, b))                      # 回し方を作っておく
            r = self.sv._ph[(PAIRS.index((si, b)), len(bp))]
            g = np.imag(np.conj(r * bp) * d).astype(np.float32)
            gs.append((tri[b], bp, g))
            nnz += int(np.count_nonzero(bp))
            sq += float(np.sum(g * g))
        if sq <= 0 or nnz == 0:
            return
        step = self.phase_lr * np.sqrt(nnz) / np.sqrt(sq)
        for word, bp, g in gs:
            self.theta[word] = step * g if word not in self.theta else self.theta[word] + step * g
            self.sv._b[word] = (bp * np.exp(1j * step * g)).astype(np.complex64)      # 長さは変わらない
        self.n_turn += 1

    def learn(self, tri):
        t = tuple(tri)
        for si, names, g, s, aim, V, q in self._asked:
            self._turn(si, names, s, aim, V, t, t[si])
        self.hear_seen(t)
        super().learn(tri)

    def spread(self):
        out = super().spread() or {}
        mv = [float(np.mean(np.abs(th[np.nonzero(th)]))) for th in self.theta.values() if np.count_nonzero(th)]
        out.update(位相を動かした語=len(mv), 位相のずれの平均=round(float(np.mean(mv)), 4) if mv else 0.0, 位相を直した回数=self.n_turn)
        return out
