"""I49f: 違和感 → 内面の波（持ち主の案、未確定）。まず 1 本目の矢印だけ。付けなければ今までと同じ動き。

持ち主の案の流れ:
  入力 → 内面波×連想 → 違和感 → 内面波へちょっと反映 → 介入 → 内面波へちょっと影響 → 違和感 → 問題なければ出力、違和感ありなら 内面波×連想 へ戻る
今ある物（I49d）: 入力・連想・違和感・出力か戻るか、までは回っている。違和感は波で持つだけで、内面の波（器）には返していない。
ここで足す物: 違和感 → 器 の矢印 1 本。介入 → 器 の矢印は、まだ無い。

形（器が「外れ」でしている事と同じ形にする。新しい数は置かない）:
  器は、聞いた物がいつもより外れると感度を上げる:  gain *= 1 + a * (外れ − いつもの外れ)
  同じに、違和感がいつもより強いと感度を上げる:    gain *= 1 + a * (今の違和感 − その波が持っている違和感)
  ・「いつも」は、違和感の波そのもの（速い波 = 今気になった、遅い波 = ずっと引っかかっている）。1 歩前の値を使う。
  ・違和感がいつもより弱ければ、感度は下がる（落ち着く）。
  ・違和感は向きを持たない（強さだけ）ので、器の波の向きは動かさない。動かすのは感度だけ。次に聞く物・戻す物の入り方が変わる。
  ・感度の上限と下限は、器の止め役（gain_max）をそのまま使う。

思考のループの実験用の口（attach）に付ける部品。固める（seal）と外れる。
手で置く物: 違和感の種類をまとめる時に平均を取る事（種類が増えても 1 歩の動きが大きくならないように）。
記録: 歩ごとの、波ごとの (今の違和感 − いつも) の平均と、掛けた倍率。
"""
from __future__ import annotations

import numpy as np

from i49d_two_loops import Node


class OddToGain(Node):
    """違和感がいつもより強い時、器の感度を上げる。弱い時は下げる。"""

    def __init__(self):
        self.log = []                                     # (歩, 波ごとの差の平均, 波ごとの倍率)

    def after(self, loop, rec):
        odd, v = loop.odd, loop.v
        K = len(odd.kinds)
        if K == 0 or not odd.raw:
            return
        x = np.array([float(odd.raw[-1].get(k, 0.0)) for k in odd.kinds], dtype=np.float32)
        usual = np.zeros((len(odd.a), K), dtype=np.float32)          # 1 歩前の違和感の波。新しく出た種類は 0
        if len(odd.hist) > 1:
            prev = odd.hist[-2]
            usual[:, : prev.shape[1]] = prev
        d = (x[None, :] - usual).mean(axis=1)                        # 波ごと: 今 − いつも（種類の平均）
        f = (1.0 + v.a * d).astype(np.float32)
        v.gain *= f
        np.clip(v.gain, 1.0 / v.gmax, v.gmax, out=v.gain)
        self.log.append((loop.n - 1, d.copy(), f.copy()))

    def arrays(self):
        if not self.log:
            return dict(step=np.zeros(0, dtype=np.int64), diff=np.zeros((0, 0), dtype=np.float32), factor=np.zeros((0, 0), dtype=np.float32))
        return dict(step=np.array([s for s, _, _ in self.log], dtype=np.int64),
                    diff=np.stack([d for _, d, _ in self.log]), factor=np.stack([f for _, _, f in self.log]))
