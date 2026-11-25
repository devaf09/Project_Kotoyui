"""I49e 聞き返しのあるやり取りを回す道具（tools/past/i49e_askback_run.py）の関門: いつもの選び方の比べと、数え。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, ".."))
import i49e_askback_run as AR  # noqa: E402
from test_i49e_askback import make, turn  # noqa: E402


def test_plain_resay_ignores_intent():
    seat = make()
    seat.listen("本", turn("猫", "魚"))
    calls = []

    def teacher(line):
        calls.append(list(line))
        if len(calls) == 1:
            return dict(分かった=False, 返事=turn("山", "川"), 意味=turn("子猫", "海", "小魚"))
        return dict(分かった=True, 返事=turn("空"), 意味=[])

    AR.make_plain_exchange()(seat, teacher, rounds=5).run()
    assert calls[1] != ["子猫", "小魚"]                    # 近さを使わないので、言いたかった向きに近い語を選ぶとは限らない


def test_count_reads_the_record(tmp_path):
    p = tmp_path / "r.jsonl"
    ex = dict(作品=6, ページ=1, 秒=3.0, 終わり="言う事が無くなった", 言い直しで分かった=1,
              番=[dict(番=0, 行=["猫"], 分かった=False, 言い直し=False, 近さ=None),
                 dict(番=1, 行=["子猫"], 分かった=True, 言い直し=True, 近さ={"子猫": 0.9})],
              先生=[dict(分かった=False, 返事の語=["山"], 意味の語=["子猫"]), dict(分かった=True, 返事の語=["空"], 意味の語=[])])
    p.write_text(json.dumps(ex, ensure_ascii=False) + "\n" + json.dumps({"秒": 3}) + "\n", encoding="utf-8")
    o = AR.count(str(p))
    assert o["言い直しの番"] == 1 and o["分からない→分かった"] == 1 and o["1番目の行が分かった"] == 0.0
    assert o["言い直しの語のうち先生の意味の語から"] == 1.0 and o["言い直しの語の近さの中央値"] == 0.9
