"""I49e: 物差し（連続する属性の軸）は、語を順番どおりに並べるか。数えだけ（持ち主の問い「連続性のある属性について、内部の偏りと答えから方向を想定できるものはある？」の下調べ、未確定）。

軸: 属性の板（内容語の板）の B 帯にある反対の組。両方の側が板にある物だけ。語の軸の値 = 片方の側の割合 − もう片方の側の割合
  （「大きい」の軸は 大きい・大きな と 小さい・小さな をまとめる）。
答え合わせ用の語の組（PAIRS）: 結果を見る前に、順番がだれの目にも明らかな名詞の組を軸ごとに書いた（前の語の方が、軸の前の側）。作った後は変えない。
数える物（組の順番が当たった割合、でたらめは .5。同点は別に数える）:
  a 属性の板の軸の値
  b 語の波の板で、表裏を残した測り方: Re(<語の向き, vec(前の側) − vec(後ろの側)>)
  c 比べ: b の絶対値（今の近さと同じで、表裏を捨てる）
板に数の無い語（a: 板に語が無い・B 帯が無い ／ b・c: 語の波が無い）を含む組は外し、外した数を出す。
手で置く物: 軸の組（AXES）、答え合わせの語の組（PAIRS）、名詞として引く事（無ければ固有名詞）。
python tools/past/i49e_axis_order.py --content <内容語の板.json> --wave <波の板> --out <stats.json>
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

AXES = {
    "大きい/小さい": (("大きい", "大きな"), ("小さい", "小さな")),
    "高い/低い": (("高い",), ("低い",)),
    "長い/短い": (("長い",), ("短い",)),
    "新しい/古い": (("新しい",), ("古い",)),
    "強い/弱い": (("強い",), ("弱い",)),
    "多い/少ない": (("多い",), ("少ない",)),
    "重い/軽い": (("重い",), ("軽い",)),
    "熱い/冷たい": (("熱い",), ("冷たい",)),
    "明るい/暗い": (("明るい",), ("暗い",)),
    "広い/狭い": (("広い",), ("狭い",)),
}

# 前の語の方が、軸の前の側（大きい・高い・長い …）。結果を見る前に書き、後から変えない。
PAIRS = {
    "大きい/小さい": [("象", "猫"), ("象", "鼠"), ("鯨", "魚"), ("山", "石"), ("海", "池"), ("家", "箱"), ("木", "草"), ("馬", "犬"),
                    ("牛", "兎"), ("船", "靴"), ("町", "家"), ("森", "木"), ("岩", "砂"), ("机", "鉛筆"), ("熊", "蜂"), ("川", "溝"),
                    ("城", "小屋"), ("島", "石"), ("象", "蟻"), ("体", "指")],
    "高い/低い": [("山", "丘"), ("塔", "家"), ("木", "草"), ("天井", "床"), ("屋根", "床"), ("雲", "屋根"), ("空", "地面"), ("星", "雲"),
                ("月", "雲"), ("峰", "麓"), ("頂上", "麓"), ("杉", "花"), ("柱", "椅子"), ("棚", "床"), ("崖", "浜"), ("高台", "谷"),
                ("屋上", "地下"), ("山", "谷"), ("空", "海"), ("塔", "門")],
    "長い/短い": [("川", "溝"), ("道", "廊下"), ("蛇", "蛙"), ("髪", "爪"), ("糸", "針"), ("年", "月"), ("月", "日"), ("日", "時間"),
                ("時間", "分"), ("一生", "一日"), ("旅", "散歩"), ("槍", "刀"), ("鉄道", "橋"), ("夏休み", "昼休み"), ("冬", "一日"),
                ("物語", "挨拶"), ("線路", "線"), ("帯", "紐"), ("竿", "箸"), ("尻尾", "耳")],
    "新しい/古い": [("今年", "去年"), ("明日", "昨日"), ("若葉", "枯葉"), ("新芽", "落葉"), ("現代", "昔"), ("未来", "過去"),
                  ("子供", "老人"), ("雛", "鶏"), ("子犬", "老犬"), ("電車", "馬車"), ("自動車", "牛車"), ("電話", "手紙"),
                  ("今日", "昨日"), ("春", "去年"), ("赤ん坊", "老人"), ("若者", "年寄り")],
    "強い/弱い": [("獅子", "兎"), ("鬼", "子供"), ("嵐", "風"), ("鉄", "紙"), ("熊", "鼠"), ("虎", "猫"), ("大人", "赤ん坊"),
                ("兵隊", "子供"), ("雷", "雨"), ("洪水", "雨"), ("象", "蟻"), ("力士", "子供"), ("鷹", "雀"), ("狼", "羊"),
                ("岩", "砂"), ("大将", "兵"), ("鬼", "人"), ("嵐", "雨")],
    "多い/少ない": [("星", "月"), ("砂", "岩"), ("葉", "幹"), ("民", "王"), ("兵", "将軍"), ("蟻", "象"), ("草", "木"), ("人々", "一人"),
                  ("群れ", "一匹"), ("町", "城"), ("髪", "眉"), ("米", "茶碗"), ("雨", "虹"), ("星", "太陽")],
    "重い/軽い": [("石", "羽"), ("鉄", "紙"), ("岩", "綿"), ("象", "鳥"), ("鉛", "羽"), ("牛", "鶏"), ("金", "紙"), ("机", "帽子"),
                ("船", "葉"), ("山", "雲"), ("水", "空気"), ("斧", "箸"), ("石", "葉"), ("鐘", "鈴"), ("荷物", "手紙"), ("大人", "子供"),
                ("熊", "兎"), ("本", "紙"), ("岩", "砂"), ("石", "綿")],
    "熱い/冷たい": [("火", "氷"), ("太陽", "月"), ("湯", "水"), ("夏", "冬"), ("炎", "雪"), ("釜", "井戸"), ("鍋", "氷"), ("焚き火", "霜"),
                  ("茶", "水"), ("昼", "夜"), ("南", "北"), ("砂漠", "雪"), ("日向", "日陰"), ("暖炉", "窓"), ("夏", "秋"), ("火", "水"),
                  ("太陽", "雪"), ("風呂", "川")],
    "明るい/暗い": [("昼", "夜"), ("太陽", "月"), ("朝", "晩"), ("光", "影"), ("日向", "日陰"), ("窓", "穴"), ("灯", "闇"), ("空", "洞穴"),
                  ("白", "黒"), ("雪", "炭"), ("朝日", "夕闇"), ("昼間", "夜中"), ("光", "闇"), ("太陽", "影")],
    "広い/狭い": [("海", "池"), ("野原", "庭"), ("空", "窓"), ("国", "町"), ("町", "家"), ("広場", "廊下"), ("平野", "谷"), ("湖", "井戸"),
                ("世界", "部屋"), ("部屋", "箱"), ("道", "路地"), ("川", "溝"), ("畑", "花壇"), ("草原", "道"), ("宇宙", "地球"), ("海", "川")],
}

NOUN_POS = ("名詞", "固有名詞")


def board_axes(cb):
    """語（名詞）→ 軸 → 値。B 帯が無い語は入れない。"""
    out = {}
    for w in cb:
        if w.get("pos") not in NOUN_POS or w["name"] in out:
            continue
        b = w.get("attrs", {}).get("B")
        if not b:
            continue
        out[w["name"]] = {ax: sum(b.get("性質:" + x, 0.0) for x in front) - sum(b.get("性質:" + x, 0.0) for x in back)
                          for ax, (front, back) in AXES.items()}
    return out


def axis_keys_present(cb):
    keys = set()
    for w in cb:
        keys |= set(w.get("attrs", {}).get("B", {}))
    return {ax: all(any("性質:" + x in keys for x in side) for side in sides) for ax, sides in AXES.items()}


def order_count(vals):
    """vals: [(前の語の値, 後の語の値)] → 当たり・外れ・同点の数と、当たりの割合（同点を除く ／ 同点を半分と数える）。"""
    hit = sum(1 for a, b in vals if a > b)
    miss = sum(1 for a, b in vals if a < b)
    tie = len(vals) - hit - miss
    n = len(vals)
    return dict(組=n, 当たり=hit, 外れ=miss, 同点=tie, 当たりの割合_同点を除く=round(hit / max(1, hit + miss), 4) if hit + miss else None,
                当たりの割合_同点は半分=round((hit + tie / 2) / max(1, n), 4) if n else None)


def wave_of(wv, name):
    for p in ("NOUN", "PROPN"):
        v = wv.of(name, p)
        if v is not None:
            return v
    return None


def main():
    import i49c_read_words as Rw
    ap = argparse.ArgumentParser()
    for k in ("content", "wave", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    cb = json.load(open(a.content, encoding="utf-8"))
    present = axis_keys_present(cb)
    ba = board_axes(cb)
    wv = Rw.WordVecs.load(a.wave)
    res = dict(軸が板にある=present, 軸ごと={})
    tot = {k: [] for k in ("a_属性の板", "b_波_表裏を残す", "c_波_絶対値")}
    for ax, pairs in PAIRS.items():
        if not present[ax]:
            continue
        front, back = AXES[ax]
        fv = [wv.of(x, "ADJ") for x in front if wv.of(x, "ADJ") is not None]
        bv = [wv.of(x, "ADJ") for x in back if wv.of(x, "ADJ") is not None]
        u = (np.mean(fv, axis=0) - np.mean(bv, axis=0)) if fv and bv else None
        va, vb, vc = [], [], []
        drop = dict(属性の板=0, 波=0)
        for x, y in pairs:
            if x in ba and y in ba:
                va.append((ba[x][ax], ba[y][ax]))
            else:
                drop["属性の板"] += 1
            px, py = wave_of(wv, x), wave_of(wv, y)
            if u is not None and px is not None and py is not None:
                sx, sy = np.vdot(u, px), np.vdot(u, py)
                vb.append((float(sx.real), float(sy.real)))
                vc.append((float(abs(sx)), float(abs(sy))))
            else:
                drop["波"] += 1
        res["軸ごと"][ax] = dict(書いた組=len(pairs), 外した組=drop, 軸の語に波がある=u is not None,
                               a_属性の板=order_count(va), b_波_表裏を残す=order_count(vb), c_波_絶対値=order_count(vc))
        tot["a_属性の板"] += va
        tot["b_波_表裏を残す"] += vb
        tot["c_波_絶対値"] += vc
    res["全体"] = {k: order_count(v) for k, v in tot.items()}
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
