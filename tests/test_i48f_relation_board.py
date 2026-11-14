"""I48f（関係の板の保存と読み込み、bot の口）の小さな確かめ（numpy だけで動く）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import i48f_relation_board as F  # noqa: E402
from test_i48e_relation_split import toy  # noqa: E402


def test_build_save_load_same_answers(tmp_path):
    works = toy()
    rb = F.RelationBoard.build(works[:50], D=256, store="p8", min_count=1, max_scenes=500)
    acc = rb.accuracy(works[50:], n=200)
    assert acc["名詞の後ろ（助詞）"]["top1"] > 0.95
    p = str(tmp_path / "board")
    rb.save(p)
    rb2 = F.RelationBoard.load(p)
    assert rb2.accuracy(works[50:], n=200) == acc
    S = [rb.scene("パン", head="食べる"), rb.scene("駅", head="行く")]
    assert np.allclose(rb.scoop("名", S), rb2.scoop("名", S))
    assert rb2.top_after("名", "パン", head="食べる")[0][0] == "を"
    assert rb2.top_after("名", "駅", head="行く")[0][0] == "に"
    assert rb2.top_after("述", "食べる")[0][0] == "た"
    assert rb2.band_read("名", "パン") is not None and rb2.band_read("名", "無い語") is None


def test_relation_light_roles():
    rb = F.RelationBoard.build(toy(), D=256, store="c64", min_count=1, max_scenes=500)
    rl = F.RelationLight(rb)
    rl.prepare([("パン", "食べる"), ("駅", "行く")])
    assert rl("パン", "目的語", "食べる") > 0.9 > 0.2 > rl("パン", "補語", "食べる")
    assert rl.particle("駅", "で・に", "行く") == "に"
    assert rl.lifts("駅", "行く")["で・に"] > rl.lifts("駅", "行く")["される"]
    assert rl("知らない", "主語", "これも知らない") == 1.0
