"""手で置いている値の一覧（不安点 2: 光の器で減らせるか）。

速さ・容量の都合の物（減らせない）と、分布や波から決められそうな物（減らす候補）を分けて持つ。
値は各ファイルの定数を読むので、ここを直しても動きは変わらない。
"""
import i48ab_long_grain as G
import i48ad_lookahead as AD
import i48aj_resonance as AJ
import i48ak_teach as AK
import i48at_surprise as AW

HANDS = [
    # 名前, 値, 置き場所, 役目, 種類（速さ・容量 / 減らす候補）, 置き換えの案
    ("ALPHA", G.ALPHA, "i48ab_long_grain", "数えの表のならし（加算）", "減らす候補", "波のエネルギーでそろえる（帯ごとの全体の量から）"),
    ("TOPK", G.TOPK, "i48ab_long_grain", "帯ごとに使う属性の数", "減らす候補", "属性の重さの分布の裾（にじみの床より上の物だけ）"),
    ("D", 8192, "各 main の --D", "波の長さ（帯の数）", "速さ・容量", "—"),
    ("K", AK.K, "i48ak_teach", "候補の数（今と三つの上位 K の和集合）", "減らす候補", "にじみの床の何倍鳴ったかで切る"),
    ("BETAS", AJ.BETAS, "i48aj_resonance", "板のならしの強さの候補", "減らす候補", "取りおきで選ぶ形は残し、候補の並びを無くす（連続で探す）"),
    ("R1_FLOOR", AJ.R1_FLOOR, "i48aj_resonance", "R1 の比の下限", "減らす候補", "にじみの床から"),
    ("best_w の候補", (0.0, 0.25, 0.5, 1.0), "i48ak_teach.Model.fit", "prior の強さ（証拠の段ごと）", "減らす候補", "位相のそろい具合から"),
    ("EV_BINS", AD.EV_BINS, "i48ad_lookahead", "証拠の数えの段", "減らす候補", "数えの分布の分位から"),
    ("DARK_W", AK.DARK_W, "i48ak_teach", "暗い側に足す重さ", "減らす候補", "使っていない（I48ak だけ）"),
    ("REFRESH", AK.REFRESH, "i48ak_teach", "表を作り直す間隔", "速さ・容量", "—"),
    ("BLOCK", AW.BLOCK, "i48at_surprise", "驚きの重さの束", "速さ・容量", "—"),
    ("BEAM", AJ.BEAM, "i48aj_resonance", "先読みの幅", "速さ・容量", "—"),
    ("取りおき", "2,000 文（β は 400 文）", "各 main", "β・best_w を選ぶ文", "速さ・容量", "—"),
]
