"""I48j（状況を周波数のスライドで渡す）の小さな確かめ。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48j_situation_slide as J  # noqa: E402


def test_situation_and_shift():
    assert J.situation("て+いる+た") == (1, 1, 0) and J.situation("ます") == (0, 0, 1) and J.situation("∅") == (0, 0, 0)
    assert J.shift_of((1, 0, 1), 3) == 15


def test_slide_switches_tense_keeps_politeness():
    cands = ["て+いる", "て+いる+た", "て+いる+ます", "て+いる+ます+た", "た", "∅"]
    S = [dict(w="歩く", pos="VERB", post=c) for c in cands[:4] for _ in range(3)] + [dict(w="見る", pos="VERB", post="た")]
    sb = J.SlideBand(cands, 1024).fit(S)
    s = dict(w="歩く", pos="VERB")
    assert cands[int(sb.scores(s, (1, 1, 0))[0].argmax())] == "て+いる+た"
    assert cands[int(sb.scores(s, (0, 1, 0))[0].argmax())] == "て+いる"
    assert cands[int(sb.scores(s, (1, 1, 1))[0].argmax())] == "て+いる+ます+た"
    r, src = sb.scores(s, (1, 0, 0))                       # 歩く の「た」だけは資料に無い → 品詞の帯から
    assert src == "品詞" and cands[int(r.argmax())] == "た"
