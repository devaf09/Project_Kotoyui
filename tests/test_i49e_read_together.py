"""I49e（tools/i49e_read_together.py）の関門のうち、本も先生も使わずに確かめられる物。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49e_read_together as RT  # noqa: E402


def test_pages_and_first_line():
    assert RT.pages_of([f"{i}。" for i in range(25)], 10)[2] == "20。21。22。23。24。"
    assert RT.first_line("<think>x</think>\n「雨がふるかも！」\n二行目") == "雨がふるかも！"


def test_one_sentence_and_copy_mask():
    assert RT.one_sent("雨がふるかも！") and not RT.one_sent("雨だ。風もふく。")
    assert RT.mask_copy("ねえ、山の奥深く迷い入ってたね", "小僧が山の奥深く迷い入って、") == "ねえ、（本文と同じ所）たね"
    assert RT.rate(["猫", "犬"], {"猫"}) == 0.5 and RT.rate([], {"猫"}) is None
