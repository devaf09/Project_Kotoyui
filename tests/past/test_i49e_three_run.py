"""I49e 三歩目の試し（tools/past/i49e_three_run.py）の関門のうち、先生もループも使わずに確かめられる物。"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import i49e_three_run as TR  # noqa: E402


def test_prompt_shows_kotoyui_line_and_role():
    said = [(1, dict(A1="雨かな", B1="ふるね", こ="雨ふる", A2="うん")), (2, dict(A1="晴れ", B1="いいね", A2="そうだね"))]
    m = TR.prompt(3, "本文", said, "A", "一言。")
    assert "ことゆい" in m[0]["content"] and "赤ちゃん" in m[0]["content"]
    assert "ことゆい「雨ふる」" in m[1]["content"] and m[1]["content"].count("ことゆい「") == 1


def test_model_phrase_and_rate():
    assert TR.MODEL_RE.search("「雨がふる」って言うといいよ") and TR.MODEL_RE.search("雨ってふるねと言ってみて")
    assert not TR.MODEL_RE.search("雨がふるね")
    assert TR.rate(["雨", "山"], {"雨"}) == 0.5 and TR.rate([], {"雨"}) is None


def test_askback_parse_takes_last_complete_json():
    sys.path.insert(0, os.path.join(HERE, "..", "tools"))
    import i49e_askback_compare as AB
    t = '考え {"x": 1} それから\n{"分かった": false, "意味の候補": "牛のこと", "返事": "牛のこと？"}'
    assert AB.parse(t) == dict(分かった=False, 意味の候補="牛のこと", 返事="牛のこと？")
    assert AB.parse("") is None and AB.parse('{"分かった": true}') is None
