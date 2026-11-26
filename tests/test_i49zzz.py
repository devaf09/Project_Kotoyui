"""I49zzz: 次の語が、話の中でもう出た語か、初めての語かの分け方。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzz_talk_memory_run as Z


def test_kind_of():
    assert Z.kind_of(["太郎"], "太郎") == ("続く", 1)
    assert Z.kind_of(["太郎", "花子", "花子"], "太郎") == ("1 つ前の別の語へ戻る", 2)
    assert Z.kind_of(["太郎", "次郎", "花子"], "太郎") == ("もっと前に出た語へ戻る", 1)
    assert Z.kind_of(["太郎", "花子"], "次郎") == ("初めての語", 1)
    assert Z.kind_of(["太郎"] * 9, "太郎") == ("続く", Z.RUN_MAX)


def test_count_sums_to_pairs():
    talk = [("太郎", "りんご", "買う"), ("太郎", "りんご", "食べる"), ("花子", "本", "読む"), ("太郎", "本", "借りる")]
    d = Z.count([talk])
    for seat in ("だれが", "何を", "どうした"):
        assert sum(d[seat]["全部"].values()) == 3
        assert sum(sum(r.values()) for r in d[seat]["続いた数"].values()) == 3
    assert d["だれが"]["全部"] == {"続く": 1, "1 つ前の別の語へ戻る": 1, "もっと前に出た語へ戻る": 0, "初めての語": 1}
    assert d["どうした"]["全部"]["初めての語"] == 3
