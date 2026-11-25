"""I49e: 聞き返しのあるやり取りの流れ（持ち主の案、未確定）。

持ち主の案: ことゆいが何か言う → 先生が分かるなら、そのままやり取りが終わるまで話す。
先生が分からない時は、聞き返すか、ゆっくり言いたい事を促す →「こういう意味？」と聞く → やり取りが終わるまで話す。

Exchange は薄い。席（i49e_three.Seat）と思考のループは変えずに使う。
  teacher(行)  → dict(分かった=真偽, 返事=[(原形, 品詞, ベクトル か None)], 意味=[同じ形])。外の先生（言葉のモデル）を包んだ物。
  run()        1 つのやり取りを最後まで回し、記録を返す。
    1. ことゆいが言う（席の speak）。言わなければ、そこで終わり（無理には話させない）。
       言った時の内面の向き（器の全体）を「言いたかった向き」として写しておく。
    2. 先生が答える。ことゆいは返事を聞く。分からない時は「こういう意味？」の語も聞く。
    3. 分かった時: ことゆいは、いつもの選び方で次を言う。言った時の向きを写し直す。
       分からない時: ことゆいは言い直す。この時だけ、選ぶ重さを「言いたかった向きへの近さ」に替える。
       先生の「こういう意味？」の語が言いたかった向きに近ければ、その語を繰り返す形になる。近さに切る線は置かず、順番だけに使う。
    4. 終わり: ことゆいが言わない（材料が少ない・前と同じ行）、または番の数 rounds に届いた時。
確かめ: 先生の「分かった」。言い直しで「分からない → 分かった」に変わった数を数えられる。
全部の番を記録する（行、分かったか、選んだ語それぞれの言いたかった向きへの近さ、先生の語のうち拾った数）。

手で置く物: 番の数 rounds、先生の名前。乱数は使わない。近さは |cos|（向きだけ）。
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import numpy as np


def closeness(vec, intent):
    if vec is None or intent is None:
        return None
    return float(np.abs(np.vdot(vec, intent)))


class Exchange:
    def __init__(self, seat, teacher, who="先", rounds=4):
        self.seat, self.teacher, self.who, self.rounds = seat, teacher, who, int(rounds)

    def _say(self, intent=None):
        """ことゆいの番。intent を渡すと、この番だけ、選ぶ重さを言いたかった向きへの近さに替える。"""
        th = self.seat.th
        vec_of = {it["名前"]: it["ベクトル"] for it in th.tray.vecs()}
        pick = getattr(th.builder, "pick", None)
        if intent is None or pick is None:
            return self.seat.speak(), vec_of
        keep = pick.weight_of
        pick.weight_of = lambda name: closeness(vec_of.get(name), intent)
        try:
            return self.seat.speak(), vec_of
        finally:
            pick.weight_of = keep

    def run(self):
        th = self.seat.th
        turns, end = [], None
        line, _ = self._say()
        if line is None:
            return dict(終わり="話さなかった", 番=turns, 言い直しで分かった=0)
        intent = th.v.whole()
        near = None                                        # 言い直しの番だけ: 選んだ語の、言いたかった向きへの近さ
        fixed = 0
        for r in range(self.rounds):
            ans = self.teacher(line)
            ok = bool(ans["分かった"])
            heard = list(ans["返事"]) + ([] if ok else list(ans.get("意味", [])))
            if turns and not turns[-1]["分かった"] and ok:
                fixed += 1
            turns.append(dict(番=r, 行=list(line), 分かった=ok, 言い直し=near is not None, 近さ=near,
                              先生の語=[l for l, _, _ in heard]))
            self.seat.listen(self.who, heard)
            if r == self.rounds - 1:
                end = "番の数に届いた"
                break
            nxt, vec_of = self._say(None if ok else intent)
            if nxt is None:
                end = "言う事が無くなった" if ok else "言い直せなかった"
                break
            turns[-1]["拾った先生の語"] = sum(1 for w in nxt if w in set(turns[-1]["先生の語"]))
            near = None if ok else {w: closeness(vec_of.get(w), intent) for w in nxt if w in vec_of}
            if ok:
                intent = th.v.whole()
            line = nxt
        return dict(終わり=end, 番=turns, 言い直しで分かった=fixed)
