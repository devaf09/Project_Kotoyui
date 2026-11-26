"""I49zzx: 前の 3 つ組から、次の 3 つ組を 1 つの返事として組む数え（小さな作り物の話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzx_reply_triple_run as X
from test_i49zzw import N, V, _attr, _talks


class PairBody:
    """本体の代わり（試験用）: 残りの 2 語の組の後に出た語を数えるだけ。聞いていない組には答えない。"""

    def __init__(self):
        self.n = {}

    def learn(self, tri):
        for si in range(3):
            d = self.n.setdefault((si,) + tuple(tri[j] for j in range(3) if j != si), {})
            d[tri[si]] = d.get(tri[si], 0) + 1

    def tick(self):
        pass

    def ask(self, si, q, cands):
        d = self.n.get((si,) + tuple(q[j] for j in range(3) if j != si), {})
        best = max(cands, key=lambda c: d.get(c, 0))
        return {"答え": best if d.get(best, 0) else None}


def test_fill_uses_fallback_when_body_is_silent():
    b = PairBody()
    b.learn((N("太郎"), N("りんご"), V("食べる")))
    cands = [N("りんご"), N("本"), N("太郎"), V("食べる")]
    assert X.fill(b, 1, (N("太郎"), None, V("食べる")), cands, N("本")) == (N("りんご"), True)
    assert X.fill(b, 1, (N("花子"), None, V("食べる")), cands, N("本")) == (N("本"), False)


def test_whole_reply_is_counted_and_body_makes_it_hang_together():
    d = X.run(_attr, _talks(), PairBody, ks=(0,), dims=512)["元の順"]
    n = d["比べ"]["問い"]
    assert n == 12 * 3 and d["比べ"]["本物の次が前と同じ"] == 0
    carry = d["全部引き継ぎ"]
    assert carry["だれが"] == n and carry["どうした"] == 0 and carry["3 席とも"] == 0 and carry["前の 3 つ組と同じ"] == n
    seat = d["席ごと（人は引き継ぎ）"]
    assert seat["だれが"] == n and seat["どうした"] > 0.8 * n
    body = d["動きを引いて、物は本体"]
    assert body["3 席とも"] >= seat["3 席とも"]                      # 物を本体で埋めると、丸ごとの当たりは下がらない
    assert body["前に丸ごと聞いた 3 つ組"] >= seat["前に丸ごと聞いた 3 つ組"]
    assert all(c["2 席以上"] >= c["3 席とも"] for k, c in d.items() if k != "比べ")


def test_without_body_only_three_forms():
    d = X.run(_attr, _talks(), None, ks=(0,), dims=512)["元の順"]
    assert set(d) == set(X.FORMS[:3]) | {"比べ"}
