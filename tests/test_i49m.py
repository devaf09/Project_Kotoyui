"""I49m の関門: 属性の割合を元の向きにする。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from i49k_seat_vec import SeatVec, Unbind  # noqa: E402
from i49m_attr_base import AttrBase  # noqa: E402

BOARD = [
    {"name": "猫", "pos": "名詞", "attrs": {"A": {"生き物": 1.0}, "B": {"性質:小さい": 0.6, "性質:柔らかい": 0.4}, "G": {"話題:家": 1.0}}},
    {"name": "犬", "pos": "名詞", "attrs": {"A": {"生き物": 1.0}, "B": {"性質:小さい": 0.3, "性質:強い": 0.7}, "G": {"話題:家": 1.0}}},
    {"name": "魚", "pos": "名詞", "attrs": {"A": {"食べ物": 0.5, "生き物": 0.5}, "G": {"話題:海": 1.0}}},
    {"name": "肉", "pos": "名詞", "attrs": {"A": {"食べ物": 1.0}, "G": {"話題:台所": 1.0}}},
    {"name": "石", "pos": "名詞", "attrs": {"A": {"物": 1.0}, "B": {"性質:硬い": 1.0}}},
    {"name": "食べる", "pos": "動詞", "attrs": {"A": {"動き:口": 1.0}}},
    {"name": "投げる", "pos": "動詞", "attrs": {"A": {"動き:手": 1.0}}},
    {"name": "空", "pos": "名詞", "attrs": {"A": {"場所": 1.0}}},
    {"name": "空", "pos": "形容詞", "attrs": {"B": {"性質:無い": 1.0}}},
    {"name": "無", "pos": "名詞", "attrs": {}},
]


def cos(a, b):
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_all_attributes_are_used_and_pos_picks_the_row():
    ab = AttrBase(BOARD)
    v = ab.of(("猫", "NOUN"))
    assert int(np.count_nonzero(v)) == 4 and len(v) == ab.n_attr
    assert cos(ab.of("猫"), ab.of("犬")) > cos(ab.of("猫"), ab.of("石"))
    assert ab.of("無") is None and ab.of("知らない") is None
    assert not np.allclose(ab.of(("空", "NOUN")), ab.of(("空", "ADJ")))
    assert AttrBase(BOARD, bands=["A", "B"]).n_attr < ab.n_attr


def test_unheard_word_is_answered_through_shared_attributes():
    ab = AttrBase(BOARD)
    u = Unbind(SeatVec(ab.of, weight="数", bind="位相"), online=True)
    u.learn(("猫", "魚", "食べる"))
    u.learn(("猫", "石", "投げる"))
    # 犬 は聞いていない。犬 は 猫 と属性が近いので、「犬 が ? を 食べる」でも 魚 が出る（語を鍵にした表では借りられない所）
    assert u.ask(1, ("犬", None, "食べる"), ["石", "魚"])["答え"] == "魚"


def test_run_tool():
    import i49m_attr_run as MR
    W = {(w["name"], "NOUN") for w in BOARD[:5]} | {("食べる", "VERB"), ("投げる", "VERB")}
    tri = lambda a, b, c: ((a, "NOUN"), (b, "NOUN"), (c, "VERB"))
    talks = [([tri("猫", "魚", "食べる")], W), ([tri("犬", "肉", "食べる")], W)]
    res = MR.run(lambda w: None, BOARD, talks, [tri("猫", "石", "投げる")])
    assert res["向き"]["属性の数"] == AttrBase(BOARD).n_attr
    assert {"解く・属性", "解く・属性・引く", "解く・属性・束ねる", "解く・属性（話題の帯を除く）", "表（属性の近さで借りる）"} <= set(res["白紙から"])
    assert res["白紙から"]["解く・属性"]["全体"]["問い"] == 6
