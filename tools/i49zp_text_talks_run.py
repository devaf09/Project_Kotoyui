"""I49zp: 「A が B を C する」の文が 1 行ずつ並んだテキストを、切り出しを通さずに 3 つ組の話にして、今までの形を数え直す。

持ち主の言葉:「これを手直しして使えないかな」「連番になっているから、使うときはシャッフルと抜き出しでテストおねがいね。」
材料のテキストはリポジトリに入れない（引数で渡す）。形態素の切り出し（GiNZA）は使わない。「が」「を」の位置で切るだけ。
手直し（直さずに、外すだけ）:
  全く同じ話（同じテキストの中）、「を」が 1 つでない行、「何を」に ほかの助詞（に・で・へ・と・から）が入る行、「の」でつないだ語、「〜をする」、動詞で終わらない行。
話の区切り（テキストに区切りが無いので、決まりで入れる）:
  その行の「だれが」が前の 8 行に無く、かつ前の 4 行の「だれが」が次の 4 行に 1 つも出ない所で切る。12 行で必ず切る。
  切った後、6 行に足りない話は、次の話とつなぐ。
  テキストに空行が 1 つでも有れば、人が入れた区切りと見て、空行だけで切る（上の決まりは使わない）。
  --text はカンマ区切りで複数渡せる（順につなぐ。ファイルの境目は必ず切る）。
並べ替え（乱数は使わない）: 話の順を、番号 × 黄金比 × k の小数部の小さい順に並べる（k = 1・2・3）。元の順も数える。話の中の行の順は変えない。
抜き出し: 答える前に問う流れなので、どの問いも聞く前の問い。加えて、流れの後ろ 2 割の問いだけの点も出す（8 割を聞いた後の、聞いていない話）。
形: 関係の表 ／ 解く形・属性だけ ／ 属性＋固有の位相（混ぜ 1・1,024 本） ／ ＋振幅 ／ 数えて選ぶ（解く側は属性だけ） ／ 数えて選ぶ（解く側は振幅の形）。まっさらからだけ（青空は先に聞かせない）。
引数: --text 材料 --wave 語の波 --content 内容語の板 --out 記録.json [--talks_out 話に切った物.txt]
手で置く物: 区切りの決まりの 8・4・12・6、後ろ 2 割、並べ替えの k、形の中の値（混ぜ具合、本数、直す速さ、temp）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

SEATS = ("だれが", "何を", "どうした")
PHI = (5 ** 0.5 - 1) / 2
ID1 = dict(mix=1.0, n_id=1024)
AMP = dict(phase_lr=0.0, rule="毎回", lr=.01, by_seat=True)
VERS = (("関係の表", "表", None, None), ("属性だけ", "解く", None, None), ("属性＋固有の位相", "解く", ID1, None),
        ("属性＋固有の位相＋振幅", "解く", ID1, AMP), ("数えて選ぶ", "数える", None, None),
        ("数えて選ぶ・振幅の形と", "数える", ID1, AMP))
EXTRA = re.compile(r"[にでへと]|から")
VERB_END = "うくぐすつぬぶむる"


def cut(line):
    """1 行 → ((だれが, 何を, どうした), None) か (None, 外した理由)。"""
    if "が" not in line:
        return None, "「が」が無い"
    s, rest = line.split("が", 1)
    if rest.count("を") != 1:
        return None, "「を」が無い" if "を" not in rest else "「を」が 2 つ以上"
    o, v = rest.split("を")
    if not s or not o or not v:
        return None, "語が欠ける"
    if "の" in s or "の" in o:
        return None, "「の」でつないだ語"
    if EXTRA.search(o) or EXTRA.search(s):
        return None, "ほかの助詞が入る"
    if re.search(r"(?:[にでへと]|から).*[一-龥ァ-ヶ]", v):               # 「を」の後ろに、もう 1 つ語が有る（箱に入れる、など）
        return None, "ほかの助詞が入る"
    if v == "する":
        return None, "〜をする"
    if v[-1] not in VERB_END:
        return None, "動詞で終わらない"
    return (s, o, v), None


def scenes(lines):
    """行の並び（空行は区切り）→ 話ごとの行番号の並び。"""
    subj = [l.split("が", 1)[0] if "が" in l else "" for l in lines]
    while lines and not lines[-1]:
        lines = lines[:-1]
    if any(not l for l in lines):                          # 人が空行で区切ったテキストは、空行だけで切る（決まりでは切らない）
        out, cur = [], []
        for i, l in enumerate(lines):
            if l:
                cur.append(i)
            elif cur:
                out.append(cur)
                cur = []
        return out + ([cur] if cur else [])
    out, cur = [], []
    for i, l in enumerate(lines):
        if not l:
            if cur:
                out.append((cur, True))
                cur = []
            continue
        if cur:
            lo = max(cur[0], i - 8)
            new = subj[i] not in subj[lo:i]
            cont = set(subj[max(cur[0], i - 4):i]) & set(subj[i:i + 4])
            if (new and not cont) or len(cur) >= 12:
                out.append((cur, False))
                cur = []
        cur.append(i)
    if cur:
        out.append((cur, True))
    merged, hold = [], []
    for sc, hard in out:
        hold += sc
        if len(hold) >= 6 or hard:
            merged.append(hold)
            hold = []
    if hold:
        merged.append(hold)
    return merged


def talks_of(lines):
    """→ (話の並び [(3 つ組の並び, 語の集まり, 元の行)], 外した理由ごとの数, 外した行の例)。語は (語, 品詞) の組。"""
    why, ex, talks, had = {}, {}, [], set()
    for sc in scenes(lines):
        key = tuple(lines[i] for i in sc)
        if key in had:                                     # 全く同じ話がもう 1 度出たら、外す
            why["同じ話の重なり"] = why.get("同じ話の重なり", 0) + 1
            continue
        had.add(key)
        tris, seen, src = [], set(), []
        for i in sc:
            t, w = cut(lines[i])
            if t is not None and t in seen:
                t, w = None, "同じ話の中の同じ行"
            if t is None:
                why[w] = why.get(w, 0) + 1
                ex.setdefault(w, [])
                if len(ex[w]) < 5:
                    ex[w].append(lines[i])
                continue
            seen.add(t)
            tris.append(((t[0], "NOUN"), (t[1], "NOUN"), (t[2], "VERB")))
            src.append(lines[i])
        if len(tris) >= 2:
            talks.append((tris, {w for t in tris for w in t}, src))
        elif tris:
            why["話に 1 行しか残らない"] = why.get("話に 1 行しか残らない", 0) + 1
    return talks, why, ex


def order(n, k):
    """乱数を使わない並べ替え。k = 0 は元の順。"""
    if k == 0:
        return list(range(n))
    return sorted(range(n), key=lambda i: ((i + 1) * PHI * k) % 1.0)


def tail_score(recs, share=.2):
    cut_at = int(len(recs) * (1 - share))
    t = recs[cut_at:]
    return round(sum(r[1] for r in t) / max(1, len(t)), 4)


def run(wave_of, attr_of, talks, ks=(0, 1, 2, 3)):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    import i49w_also_right_run as WR
    import i49za_fold_run as ZR
    import i49zk_mix_phase_run as ZK
    from i49zb_mix import Mix
    from i49zj_phase_read import PhaseRead
    from i49zm_two_channel import TwoChannel
    from i49zm_two_channel_run import Peek
    plain = [(t[0], t[1]) for t in talks]
    ok = WR.ok_sets([t for tris, _ in plain for t in tris])
    out = {}
    for k in ks:
        name_k = "元の順" if k == 0 else "並べ替え %d" % k
        seq = [plain[i] for i in order(len(plain), k)]
        out[name_k], keep = {}, {}
        for name, kind, two, read in VERS:
            table = RL.Relation(wave_of, half=HR.HALF)
            inner = None
            if kind != "表":
                sv = KV.SeatVec(attr_of if two is None else TwoChannel(attr_of, **two).of, weight="数", bind="位相")
                inner = PhaseRead(sv, **read) if read else KV.Unbind(sv, online=True)
            rel = table if kind == "表" else Peek(table, inner) if kind == "解く" else Mix(table, inner, upper=None)
            recs = ZR.stream(rel, seq, ok)
            d = ZR.summary(recs, 0, keep.get("関係の表"))
            d["後ろ 2 割"] = tail_score(recs)
            if kind == "解く":
                d["段ごとの当たり"] = {lv: dict(解く=h[0], 問い=h[1]) for lv, h in rel.tally.items()}
            if kind == "数える":
                d["段ごとの当たり"] = {lv: dict(表=h[0], 解く=h[1], 問い=h[2]) for lv, h in rel.tally.items()}
            keep[name] = recs
            out[name_k][name] = d
            print(name_k, name, d["正しい物"], d["合わせて"], d["後ろ 2 割"], flush=True)
        out[name_k]["直の比べ"] = {"%s → %s" % (a, b): ZK.duel(keep[a], keep[b]) for a, b in
                                (("属性だけ", "属性＋固有の位相"), ("関係の表", "属性＋固有の位相"), ("関係の表", "数えて選ぶ"), ("関係の表", "属性だけ"),
                                 ("関係の表", "属性＋固有の位相＋振幅"), ("属性だけ", "属性＋固有の位相＋振幅"), ("属性＋固有の位相＋振幅", "数えて選ぶ"), ("数えて選ぶ", "数えて選ぶ・振幅の形と"))}
    return out


def cover(talks, wave_of, attr_of):
    """語の波・属性を持たない語の数（種類と、行に占める割合）。"""
    words = {w for _, ws, _ in talks for w in ws}
    no_w = {w for w in words if wave_of(w) is None}
    no_a = {w for w in words if attr_of(w) is None}
    n = max(1, sum(len(t[0]) for t in talks))
    return dict(語の種類=len(words), 波が無い語=len(no_w), 属性が無い語=len(no_a),
                波が無い語を含む行の割合=round(sum(any(w in no_w for w in tri) for t in talks for tri in t[0]) / n, 4),
                属性が無い語を含む行の割合=round(sum(any(w in no_a for w in tri) for t in talks for tri in t[0]) / n, 4),
                席ごとの属性が無い行=[sum(tri[i] in no_a for t in talks for tri in t[0]) for i in range(3)])


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--talks_out")
    a = ap.parse_args()
    t0 = time.time()
    talks, why, n_lines = [], {}, 0
    for f in a.text.split(","):
        lines = [l.strip() for l in open(f, encoding="utf-8").read().splitlines()]
        t, w, _ = talks_of(lines)
        talks += t
        n_lines += sum(1 for l in lines if l)
        for k, v in w.items():
            why[k] = why.get(k, 0) + v
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    if a.talks_out:
        with open(a.talks_out, "w", encoding="utf-8") as f:
            for i, (tris, _, src) in enumerate(talks):
                f.write("# 話 %d\n%s\n\n" % (i, "\n".join(src)))
    sizes = [len(t[0]) for t in talks]
    res = dict(行=n_lines, 話=len(talks), 三つ組=sum(sizes), 話ごとの三つ組の平均=round(sum(sizes) / max(1, len(sizes)), 2),
               話ごとの語の平均=round(sum(len(t[1]) for t in talks) / max(1, len(talks)), 2), 外した数=why, 語の持ち物=cover(talks, wave_of, full.of))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    res.update(run(wave_of, full.of, talks))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
