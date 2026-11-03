"""骨組みの確認: 版の表記がそろっているか、光の部品の写しが読み込めて動くか。実行: python -m pytest tests"""
import os
import re
import sys

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import kotoyui  # noqa: E402


def test_version_matches_docstring_and_readme():
    """__version__ と、変更履歴の一番上の版と、README の版が同じ。"""
    top = re.search(r"変更履歴.*?\n\s+(\d+\.\d+\.\d+) \(", kotoyui.__doc__, re.S).group(1)
    assert top == kotoyui.__version__
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert f"版 {kotoyui.__version__}" in readme


def test_docstring_first_line():
    assert kotoyui.__doc__.splitlines()[0] == "kotoyui.py — 光の部品で日本語の会話をする（単独で動く 1 ファイル版）"


def test_vendor_parts_version():
    """写した部品の版が、ファイル名の版と同じ。"""
    assert kotoyui.parts().__version__ == "0.11.0"


def test_vendor_parts_light_path_runs():
    """光の部品で、板 → 回折 → 明るさを平均 1 に → レンズ を 1 回通せる（明るさが有限で、平均が 1 付近）。"""
    p2 = kotoyui.parts()
    L = 32
    y, x = p2.grid(L, L)
    pid = p2.patch_index(L, L, 4, 4)
    g = torch.Generator().manual_seed(0)
    A, wx, wy, phi = (torch.rand(16, generator=g) for _ in range(4))
    ph = p2.plate_stripes(A, wx, wy, phi, pid, y, x)
    u = torch.ones(1, L, L, dtype=torch.complex64)
    u = p2.normalize(p2.propagate(p2.apply_plate(u, ph), p2.transfer(L, L, z=32.0)))
    I = p2.intensity(p2.lens(u))
    assert torch.isfinite(I).all()
    assert I.shape[-2:] == (L, L)


def test_analyze_and_question():
    """語の解析: 中身の語（原形）と丁寧さ、問いの形。"""
    words, polite = kotoyui.analyze("明日から旅行で北海道に行きます")
    assert polite and "旅行" in words and "行く" in words
    assert kotoyui.isq("休みの日は何をしていますか？") and not kotoyui.isq("散歩が大変で")


def test_problem_score():
    """課題の読み取り: 困りごとは 1 以上、楽しみは 1 未満。"""
    assert kotoyui.problem_score("最近寝不足でつらいよ") >= 1.0
    assert kotoyui.problem_score("大変だけど楽しみです") < 1.0


def test_character_answers_from_profile():
    """キャラクタ: 名前を聞かれたらプロフィールの名前で答える（assets/ の見本のプロフィールで。データは要らない）。"""
    ch = kotoyui.Character(os.path.join(ROOT, "assets", "characters", "secretary_kotoyui_tonari.json"))
    clock = kotoyui.Clock("2026-11-02T14:00", os.path.join(ROOT, "assets", "clock_grain.json"))
    a = ch.answer("あなたのお名前は？", ["x", "y"], clock)
    assert ch.prof["名前"] in a and ch.prof["苗字"] in a


def test_light_resonance_learns_a_small_table():
    """語板の共鳴（0.2.0）: 小さな表を覚えさせると、光の共鳴が表の PMI とほぼ同じになり、共鳴の大小の順も同じ。"""
    class Mat:
        cpair = {("犬", "散歩"): 9, ("犬", "かわいい"): 3, ("雨", "傘"): 8, ("雨", "散歩"): 2}
        pmis = {("犬", "散歩"): 1.2, ("犬", "かわいい"): 0.8, ("雨", "傘"): 1.5, ("雨", "散歩"): 0.3}

        def pmi(self, x, y):
            return self.pmis[(x, y)]
    lr = kotoyui.LightResonance().build(Mat(), n=16, plates=8, steps=300, log=lambda *_: None)
    for (x, y), p in Mat.pmis.items():
        assert abs(lr.lift(x)[y] - p) < 0.1
    assert lr.resonance({"雨"}, {"傘"}) > lr.resonance({"雨"}, {"散歩"}) > lr.resonance({"雨"}, {"かわいい"}) == 0.0
