"""I49zzh: 害の問いの抜き出し（小さな世界）。"""
import i49_foundation as FD
import i49zzh_harm_examples as H
from test_i49zw import attr as small
from test_i49zzd import _talks


def test_pick_is_spread_not_the_head():
    p = H.pick(100, 20)
    assert len(p) == 20 == len(set(p)) and p == sorted(p) and max(p) > 50 and H.pick(5, 20) == [0, 1, 2, 3, 4]


def test_peek_keeps_the_last_answer_without_changing_it():
    get = lambda w: small(w[0])
    b, p = FD.body(get, get), H.Peek(FD.body(get, get))
    tri = (("犬", "NOUN"), ("骨", "NOUN"), ("かじる", "VERB"))
    b.learn(tri)
    p.learn(tri)
    q = (tri[0], tri[1], None)
    assert p.ask(2, q, [tri[2], ("読む", "VERB")]) == b.ask(2, q, [tri[2], ("読む", "VERB")]) and p.out["答え"] == tri[2]


def test_collect_counts_match_the_counting_tool():
    import i49zx_teach_board_run as R
    get = lambda w: small(w[0])
    talks = _talks()
    seq = [(t[0], t[1]) for t in talks]
    harm, up, n = H.collect(FD.body(get, get), FD.body(get, get), seq[:len(seq) // 2], seq[len(seq) // 2:])
    s = R.run_unheard(get, get, talks, rates=(1.0,), ks=(0,), dim=1024, ext=True)["元の順"]
    assert n == s["本体だけ（後半を聞かない）"]["問い"]
    assert sum(c["外れに変わった"] for c in s["介入 100%"]["外付けを使った段の組"].values()) == len(harm)
    assert sum(c["当たりに変わった"] for c in s["介入 100%"]["外付けを使った段の組"].values()) == up
    assert all(h["本体"]["答え"] == h["正解"] != h["外付け"]["答え"] for h in harm)
