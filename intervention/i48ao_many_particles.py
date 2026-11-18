"""I48ao: 助詞をまとめて介入で教える（I48an の続き）。「と」だけ教えると「と」が漏れるので、ほかの助詞も一緒に教えた時に漏れが抑えられるかを見る。

- 介入の板: P＋Q（並びの板と属性の板、Q0 同じ重さ）。本体には融合しない。教えた記録は JSONL に残す。
- 役の合わせ方（--hows）: 掛け算（役の点の和）、足し算（log 平均 exp）、中間（教えた板が鳴った役だけの点の平均 = 鳴った役の数で割る）。
- 教える文: 全部自作（GiNZA を使わず、要点と詰め物を手で書く）。と 18（I48al の基本 10・軸1 5・軸3 3）、で 10、に 10、を 10、が 10、へ・から・も 4 ずつ。計 70。
- 教える数を 0 → 10 → 30 → 全部 と増やす。教える順は乱数で 3 通り（seed 0・1・2）。各段で測る。
- 測る物: 間・文末・文、助詞（と・で・に・を・が・へ・から・も）ごとの当たり（元がその助詞の間で、その助詞を選んだ割合）と害（元がその助詞でない間を
  その助詞にした数）、広げたい例 5 つと広げたくない例 4 つの「と」の順位（I48am と同じ）。
- --one: 1 文だけ教えた時の数字（全学習との比べ用）。
- 区切りごとに JSON に書き出す（途中で止まっても残る）。
手で置く物: 教える文（TEACH_ALL）、段（10・30・全部）、順の数 3。
python intervention/i48ao_many_particles.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--hows 掛け算 足し算 中間] [--one 文]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48al_teach_one as AL  # noqa: E402
import i48am_attr_weight as AM  # noqa: E402
import i48an_attr_board as AN  # noqa: E402

N, V, P_ = "NOUN", "VERB", "PRON"
PARTICLES = ("と", "で", "に", "を", "が", "へ", "から", "も")


def s2(a, ua, b, p, end="た。"):
    """「a p b した」形の 2 語の自作文。"""
    return [(a, ua), (b, V)], [p, end]


MADE = [
    ("で", "公園で遊んだ", *s2("公園", N, "遊ぶ", "で", "だ。")),
    ("で", "学校で勉強した", [("学校", N), ("勉強", N), ("する", V)], ["で", "", "た。"]),
    ("で", "家で食べた", *s2("家", N, "食べる", "で")),
    ("で", "図書館で本を読んだ", [("図書館", N), ("本", N), ("読む", V)], ["で", "を", "だ。"]),
    ("で", "駅で待った", *s2("駅", N, "待つ", "で")),
    ("で", "店で買った", *s2("店", N, "買う", "で")),
    ("で", "海で泳いだ", *s2("海", N, "泳ぐ", "で", "だ。")),
    ("で", "部屋で寝た", *s2("部屋", N, "寝る", "で")),
    ("で", "会社で働いた", *s2("会社", N, "働く", "で")),
    ("で", "バスで来た", *s2("バス", N, "来る", "で")),
    ("に", "学校に行った", *s2("学校", N, "行く", "に")),
    ("に", "友達に会った", *s2("友達", N, "会う", "に")),
    ("に", "机に置いた", *s2("机", N, "置く", "に")),
    ("に", "町に住んだ", *s2("町", N, "住む", "に", "だ。")),
    ("に", "先生に聞いた", *s2("先生", N, "聞く", "に")),
    ("に", "駅に着いた", *s2("駅", N, "着く", "に")),
    ("に", "母に電話した", [("母", N), ("電話", N), ("する", V)], ["に", "", "た。"]),
    ("に", "部屋に入った", *s2("部屋", N, "入る", "に")),
    ("に", "妹に本を貸した", [("妹", N), ("本", N), ("貸す", V)], ["に", "を", "た。"]),
    ("に", "箱に入れた", *s2("箱", N, "入れる", "に")),
    ("を", "本を読んだ", *s2("本", N, "読む", "を", "だ。")),
    ("を", "パンを食べた", *s2("パン", N, "食べる", "を")),
    ("を", "映画を見た", *s2("映画", N, "見る", "を")),
    ("を", "手紙を書いた", *s2("手紙", N, "書く", "を")),
    ("を", "水を飲んだ", *s2("水", N, "飲む", "を", "だ。")),
    ("を", "窓を開けた", *s2("窓", N, "開ける", "を")),
    ("を", "歌を歌った", *s2("歌", N, "歌う", "を")),
    ("を", "車を買った", *s2("車", N, "買う", "を")),
    ("を", "道を歩いた", *s2("道", N, "歩く", "を")),
    ("を", "宿題をした", *s2("宿題", N, "する", "を")),
    ("が", "犬が走った", *s2("犬", N, "走る", "が")),
    ("が", "雨が降った", *s2("雨", N, "降る", "が")),
    ("が", "花が咲いた", *s2("花", N, "咲く", "が")),
    ("が", "電車が来た", *s2("電車", N, "来る", "が")),
    ("が", "子供が泣いた", *s2("子供", N, "泣く", "が")),
    ("が", "鳥が飛んだ", *s2("鳥", N, "飛ぶ", "が", "だ。")),
    ("が", "風が吹いた", *s2("風", N, "吹く", "が")),
    ("が", "猫が寝た", *s2("猫", N, "寝る", "が")),
    ("が", "先生が来た", *s2("先生", N, "来る", "が")),
    ("が", "星が光った", *s2("星", N, "光る", "が")),
    ("へ", "学校へ行った", *s2("学校", N, "行く", "へ")),
    ("へ", "公園へ行った", *s2("公園", N, "行く", "へ")),
    ("へ", "海へ行った", *s2("海", N, "行く", "へ")),
    ("へ", "家へ帰った", *s2("家", N, "帰る", "へ")),
    ("から", "家から来た", *s2("家", N, "来る", "から")),
    ("から", "駅から歩いた", *s2("駅", N, "歩く", "から")),
    ("から", "学校から帰った", *s2("学校", N, "帰る", "から")),
    ("から", "友達から聞いた", *s2("友達", N, "聞く", "から")),
    ("も", "私も行った", *s2("私", P_, "行く", "も")),
    ("も", "犬も走った", *s2("犬", N, "走る", "も")),
    ("も", "母も食べた", *s2("母", N, "食べる", "も")),
    ("も", "彼も来た", *s2("彼", P_, "来る", "も")),
]
TEACH_ALL = [("と", t, k, g) for a, t, k, g in AL.TEACH_SET if a in ("基本", "軸1", "軸3")] + MADE
STAGES = (10, 30, None)


class OModel(AN.QModel):
    def q_score(self, keys, i):
        if self.how != "中間":
            return super().q_score(keys, i)
        if not self.Lq:
            return 0
        parts = []
        for r, key in AN.role_keys(keys, i).items():
            if key is None or r not in self.Lq:
                continue
            f, v = self.S.attrs(*key)
            parts.append(self.Lq[r][:, f] @ (v / max(1e-9, float(v.sum()))))
        if not parts:
            return 0
        P = np.stack(parts)
        return P.sum(0) / np.maximum(1, (P != 0).sum(0))       # 鳴った役だけの点の平均


def measure(m, te, surf):
    c = {}

    def inc(k, v=1):
        c[k] = c.get(k, 0) + v
    for s in te:
        xs = AE.items(m.S, s, "節")
        if not xs:
            continue
        cont, gaps, head = G.split(s)
        rs, _ = m.walk(AK.keys_of(s), xs)
        seq = [m.pg.cls[r["選び"]] for r in rs]
        inc("文")
        for i, (g, y) in enumerate(zip(seq, gaps)):
            if i + 1 == len(gaps):
                inc("文末当たり", g == y)
                continue
            inc("間")
            inc("間当たり", g == y)
            for p in PARTICLES:
                if y == p:
                    inc(p + "n")
                    inc(p + "当たり", g == p)
                elif g == p:
                    inc(p + "害")
        sent = "".join(surf.form(l, u, g) + g for (l, f_, u), g in zip(cont, seq))
        inc("文当たり", G.norm(head + sent) == G.norm(s["text"]))
    d = lambda a, b: round(c.get(a, 0) / max(1, c.get(b, 0)), 4)
    return dict(間=d("間当たり", "間"), 文末=d("文末当たり", "文"), 文=d("文当たり", "文"),
                助詞={p: dict(数=c.get(p + "n", 0), 当たり=d(p + "当たり", p + "n"), 害=c.get(p + "害", 0)) for p in PARTICLES},
                例={k: v["順位"] for k, v in AM.ranks_of(m).items()})


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    ap.add_argument("--hows", nargs="*", default=["掛け算", "足し算", "中間"])
    ap.add_argument("--orders", type=int, default=3)
    ap.add_argument("--one", default="友達と散歩に行った")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill = AD.Fill(S)
    surf = G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    base = OModel(S, fill, a.r1).fit(tr, hold).init_q()
    missing = sorted({g for *_, gs in TEACH_ALL for g in gs if g not in base.ci})
    res = dict(学びの文=len(tr), 試しの文=len(te), R1の型=a.r1, 教える文の数=len(TEACH_ALL), 詰め物の組に無い=missing, 土台=measure(base, te, surf), 型={})
    out_dir = os.path.dirname(os.path.abspath(a.out))
    stem = os.path.splitext(os.path.basename(a.out))[0]

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"学び {time.time() - t0:.0f} 秒", json.dumps(res["土台"], ensure_ascii=False), "組に無い", missing, flush=True)
    save()
    items = {it[1]: it for it in TEACH_ALL}
    for how in a.hows:
        o = res["型"][how] = {}
        if a.one:
            t1 = time.time()
            m = base.clone()
            AN.teach(m, items[a.one], ("P+Q", "Q0", how), None)
            o["1 文 " + a.one] = measure(m, te, surf)
            print(how, "1 文", round(time.time() - t1), "秒", json.dumps(o["1 文 " + a.one], ensure_ascii=False), flush=True)
            save()
        for seed in range(a.orders):
            order = [TEACH_ALL[k] for k in np.random.default_rng(seed).permutation(len(TEACH_ALL))]
            m = base.clone()
            done = 0
            o[f"順 {seed}"] = {}
            for st in STAGES:
                t1 = time.time()
                upto = len(order) if st is None else st
                for it in order[done:upto]:
                    AN.teach(m, it, ("P+Q", "Q0", how), None)
                done = upto
                o[f"順 {seed}"][str(upto)] = r = measure(m, te, surf)
                print(how, "順", seed, upto, "文", round(time.time() - t1), "秒", json.dumps(r, ensure_ascii=False), flush=True)
                save()
            o[f"順 {seed}"]["教えた順"] = [it[1] for it in order]
            m.save_log(os.path.join(out_dir, f"{stem}_{how}_順{seed}_記録.jsonl"))
            save()
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
