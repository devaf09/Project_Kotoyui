"""I49zr: 席を増やした枠（が・を・に・で・と・へ・から・どうした）で、表・波で解く形・数えて選ぶを数える（次の段階の 1 つ目）。

材料は tools/i49zp_text_talks_run.py と同じテキスト（リポジトリに入れない。--text はカンマ区切りで複数）。形態素の切り出しは使わない。
行の切り方（決まりで切るだけ。直さない）:
  初めの「が」の前が「が」の席。残りを、助詞（を・に・で・と・へ・から）で切る。最後の残りが「どうした」。
  助詞と見るのは、すぐ前がひらがなでなく、後ろに 2 字より多く残る時だけ（「撫でる」「落とす」「おにぎり」を切らないため）。「を」はいつも助詞。
  外す行: 「が」が無い、「の」「と」でつないだ語、同じ助詞が 2 回、どうした が動詞で終わらない。
  分かっている粗さ: 「静かに」などの様子の語は どうした の中に残る。「丁寧に」のような漢字の様子の語は「に」の席に入る。
3 通りに数える（どれも、答える前に問う。候補 = 同じ話の語から、見せた語を除いた物）:
  三つの席  が・を・どうした だけの行（I49zp と同じ行）。今までの関係の表（I49e）・解く形（I49k）も並べて、作り直した部品と比べる。
  全部の席  切れた行を全部。隠す席ごと・見せた席の数ごとの当たりも出す。
  席を見せない  全部の席と同じ行を覚えるが、問う時に が・を・どうした 以外の席を見せない（同じ問いで、足した席が効いたかを見る）。
形: 表（FrameTable） ／ 解く形・属性（FrameSolve） ／ 数えて選ぶ（Mix、手で置く段は無し）。
    I49zs: 表・席ごと（FrameSeatTable）と、それを使う 数えて選ぶ・席ごと も並べる。
話の並べ方: 元の順と、乱数を使わない並べ替え 3 通り。
引数: --text 材料 --wave 語の波 --content 内容語の板 --out 記録.json [--frames_out 枠に切った物.txt]
手で置く物: 席の並び、助詞と見る決まり、並べ替えの k、後ろ 2 割。
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

import i49zp_text_talks_run as ZP  # noqa: E402
from i49zq_frame import SEATS  # noqa: E402

CORE = (SEATS.index("が"), SEATS.index("を"), SEATS.index("どうした"))
PART = re.compile(r"から|[をにでとへ]")
HIRA = re.compile(r"[ぁ-ん]")
VERB_END = "うくぐすつぬぶむる"


def cut(line):
    """1 行 → (枠, None) か (None, 外した理由)。枠の語は (語, 品詞) の組。"""
    if "が" not in line:
        return None, "「が」が無い"
    s, rest = line.split("が", 1)
    got, pos = {"が": s}, 0
    for m in PART.finditer(rest):
        i = m.start()
        if i <= pos:
            continue
        p = m.group()
        if p != "を" and (HIRA.match(rest[i - 1]) or len(rest) - m.end() < 2):
            continue                                       # 「撫でる」「落とす」の送り仮名は助詞にしない
        if p in got:
            return None, "同じ助詞が 2 回"
        got[p] = rest[pos:i]
        pos = m.end()
    v = rest[pos:]
    if not s or not v or any(not w for w in got.values()):
        return None, "語が欠ける"
    if v[-1] not in VERB_END:
        return None, "動詞で終わらない"
    for w in got.values():
        if re.search(r"[一-龥ァ-ヶー][のと]", w) or "を" in w:
            return None, "「の」「と」でつないだ語"
    f = [None] * len(SEATS)
    for p, w in got.items():
        f[SEATS.index(p)] = (w, "NOUN")
    f[-1] = (v, "VERB")
    return tuple(f), None


def talks_of(lines):
    """→ (話の並び [(枠の並び, 語の集まり, 元の行)], 外した理由ごとの数, 外した行の例)。"""
    why, ex, talks, had = {}, {}, [], set()
    for sc in ZP.scenes(lines):
        key = tuple(lines[i] for i in sc)
        if key in had:
            why["同じ話の重なり"] = why.get("同じ話の重なり", 0) + 1
            continue
        had.add(key)
        frames, seen, src = [], set(), []
        for i in sc:
            f, w = cut(lines[i])
            if f is not None and f in seen:
                f, w = None, "同じ話の中の同じ行"
            if f is None:
                why[w] = why.get(w, 0) + 1
                if len(ex.setdefault(w, [])) < 5:
                    ex[w].append(lines[i])
                continue
            seen.add(f)
            frames.append(f)
            src.append(lines[i])
        if len(frames) >= 2:
            talks.append((frames, {w for f in frames for w in f if w is not None}, src))
    return talks, why, ex


def is_core(f):
    return all(f[j] is not None for j in CORE) and all(w is None for j, w in enumerate(f) if j not in CORE)


def strip(f):
    """が・を・どうした 以外の席を見せない枠。"""
    return tuple(w if j in CORE else None for j, w in enumerate(f))


def ok_sets(talks):
    ok = {}
    for frames, _, _ in talks:
        for f in frames:
            for si, w in enumerate(f):
                if w is not None:
                    ok.setdefault((si, tuple(None if j == si else x for j, x in enumerate(f))), set()).add(w)
    return ok


def stream(rel, talks, ok, show=lambda f: f):
    """→ 問いごとの (隠した席, 当たり, 正しいと言える物, 見せた席の数, 足した席が有る枠か, 問いの名札)。show = 問う時に見せる枠の作り方。"""
    recs = []
    for ti, (frames, words, _) in enumerate(talks):
        for fi, f in enumerate(frames):
            q = show(f)
            for si, w in enumerate(q):
                if w is None:
                    continue
                shown = {x for j, x in enumerate(q) if j != si and x is not None}
                if not shown:
                    continue
                cands = sorted({x for x in words if x not in shown} | {w}, key=str)
                if len(cands) < 2:
                    continue
                a = rel.ask(si, tuple(None if j == si else x for j, x in enumerate(q)), cands)["答え"]
                r = a == w
                good = r or (a is not None and a in ok.get((si, tuple(None if j == si else x for j, x in enumerate(f))), ()))
                recs.append((si, r, good, len(shown), not is_core(f), (ti, fi, si)))
            rel.learn(f)
        rel.tick()
    return recs


def summary(recs):
    def rate(rs, k=1):
        return round(sum(r[k] for r in rs) / len(rs), 4) if rs else None
    out = dict(問い=len(recs), 正しい物=rate(recs), 合わせて=rate(recs, 2), 後ろ2割=ZP.tail_score(recs))
    out["席ごと"] = {SEATS[i]: dict(問い=sum(r[0] == i for r in recs), 正しい物=rate([r for r in recs if r[0] == i])) for i in range(len(SEATS))
                  if any(r[0] == i for r in recs)}
    out["見せた席の数ごと"] = {str(n): dict(問い=sum(r[3] == n for r in recs), 正しい物=rate([r for r in recs if r[3] == n])) for n in sorted({r[3] for r in recs})}
    ext = [r for r in recs if r[4]]
    out["足した席が有る枠の問い"] = dict(問い=len(ext), 正しい物=rate(ext),
                                   核の席=rate([r for r in ext if r[0] in CORE]), 足した席=rate([r for r in ext if r[0] not in CORE]))
    return out


def duel(a, b):
    up = sum(1 for x, y in zip(a, b) if not x[1] and y[1])
    dn = sum(1 for x, y in zip(a, b) if x[1] and not y[1])
    return dict(良くなった=up, 悪くなった=dn, z=round((up - dn) / max(1.0, (up + dn) ** 0.5), 2))


def forms(wave_of, attr_of):
    import i49zq_frame as Q
    from i49zb_mix import Mix
    return (("表", lambda: Q.FrameTable(wave_of)), ("解く形", lambda: Q.FrameSolve(Q.FrameVec(attr_of))),
            ("数えて選ぶ", lambda: Mix(Q.FrameTable(wave_of), Q.FrameSolve(Q.FrameVec(attr_of)), upper=None)),
            ("表・席ごと", lambda: Q.FrameSeatTable(wave_of)),
            ("数えて選ぶ・席ごと", lambda: Mix(Q.FrameSeatTable(wave_of), Q.FrameSolve(Q.FrameVec(attr_of)), upper=None)))


class Tri:
    """今までの 3 つ組の部品（関係の表・Unbind・Mix）を、枠の形で呼ぶための包み。が・を・どうした だけの枠に使う。"""

    def __init__(self, inner):
        self.inner = inner

    def ask(self, si, frame, cands):
        return self.inner.ask(CORE.index(si), tuple(frame[j] for j in CORE), cands)

    def learn(self, frame):
        self.inner.learn(tuple(frame[j] for j in CORE))

    def tick(self):
        self.inner.tick()


def old_forms(wave_of, attr_of):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    from i49zb_mix import Mix

    def solve():
        return KV.Unbind(KV.SeatVec(attr_of, weight="数", bind="位相"), online=True)
    return (("今までの表", lambda: Tri(RL.Relation(wave_of, half=HR.HALF))), ("今までの解く形", lambda: Tri(solve())),
            ("今までの数えて選ぶ", lambda: Tri(Mix(RL.Relation(wave_of, half=HR.HALF), solve(), upper=None))))


def run(wave_of, attr_of, talks, ks=(0, 1, 2, 3)):
    ok = ok_sets(talks)
    core = [([f for f in fr if is_core(f)], ws, src) for fr, ws, src in talks]
    core = [(fr, {w for f in fr for w in f if w is not None}, src) for fr, ws, src in core if len(fr) >= 2]
    ok_core = ok_sets(core)
    out = {}
    for k in ks:
        name_k = "元の順" if k == 0 else "並べ替え %d" % k
        d, keep = {}, {}
        seq = [core[i] for i in ZP.order(len(core), k)]
        d["三つの席"] = {}
        for name, make in forms(wave_of, attr_of) + old_forms(wave_of, attr_of):
            keep[("三", name)] = stream(make(), seq, ok_core)
            d["三つの席"][name] = summary(keep[("三", name)])
        seq = [talks[i] for i in ZP.order(len(talks), k)]
        d["全部の席"], d["席を見せない"] = {}, {}
        for name, make in forms(wave_of, attr_of):
            keep[("全", name)] = stream(make(), seq, ok)
            d["全部の席"][name] = summary(keep[("全", name)])
            keep[("隠", name)] = stream(make(), seq, ok, show=strip)
            d["席を見せない"][name] = summary(keep[("隠", name)])
            print(name_k, name, d["三つの席"][name]["正しい物"], d["全部の席"][name]["正しい物"], d["席を見せない"][name]["正しい物"], flush=True)
        # 同じ問い（足した席が有る枠の、が・を・どうした を隠す問い）で、足した席を見せた時と見せない時
        d["足した席を見せると"] = {}
        for name, _ in forms(wave_of, attr_of):
            b = {r[5]: r for r in keep[("全", name)] if r[4] and r[0] in CORE}
            a = [r for r in keep[("隠", name)] if r[4] and r[5] in b]
            b = [b[r[5]] for r in a]
            if a:
                d["足した席を見せると"][name] = dict(問い=len(a), 見せない=round(sum(r[1] for r in a) / max(1, len(a)), 4),
                                                 見せる=round(sum(r[1] for r in b) / max(1, len(b)), 4), 直の比べ=duel(a, b))
        d["直の比べ"] = {"三つの席｜今までの表 → 表": duel(keep[("三", "今までの表")], keep[("三", "表")]),
                      "三つの席｜今までの解く形 → 解く形": duel(keep[("三", "今までの解く形")], keep[("三", "解く形")]),
                      "三つの席｜今までの数えて選ぶ → 数えて選ぶ": duel(keep[("三", "今までの数えて選ぶ")], keep[("三", "数えて選ぶ")]),
                      "三つの席｜今までの表 → 表・席ごと": duel(keep[("三", "今までの表")], keep[("三", "表・席ごと")]),
                      "三つの席｜今までの数えて選ぶ → 数えて選ぶ・席ごと": duel(keep[("三", "今までの数えて選ぶ")], keep[("三", "数えて選ぶ・席ごと")]),
                      "全部の席｜表 → 表・席ごと": duel(keep[("全", "表")], keep[("全", "表・席ごと")]),
                      "全部の席｜数えて選ぶ → 数えて選ぶ・席ごと": duel(keep[("全", "数えて選ぶ")], keep[("全", "数えて選ぶ・席ごと")]),
                      "全部の席｜表 → 数えて選ぶ": duel(keep[("全", "表")], keep[("全", "数えて選ぶ")]),
                      "全部の席｜表 → 解く形": duel(keep[("全", "表")], keep[("全", "解く形")])}
        out[name_k] = d
    return out


def main():
    import i49c_read_words as Rw
    import i49m_attr_base as AB
    ap = argparse.ArgumentParser()
    for k in ("text", "wave", "content", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--frames_out")
    a = ap.parse_args()
    t0 = time.time()
    talks, why = [], {}
    for f in a.text.split(","):
        lines = [l.strip() for l in open(f, encoding="utf-8").read().splitlines()]
        t, w, _ = talks_of(lines)
        talks += t
        for k, v in w.items():
            why[k] = why.get(k, 0) + v
    wv = Rw.WordVecs.load(a.wave)

    def wave_of(w):
        v = wv.of(*w)
        return wv.of(w[0], "PROPN") if v is None and w[1] == "NOUN" else v

    full = AB.AttrBase(json.load(open(a.content, encoding="utf-8")))
    if a.frames_out:
        with open(a.frames_out, "w", encoding="utf-8") as f:
            for i, (frames, _, src) in enumerate(talks):
                f.write("# 話 %d\n" % i)
                for fr, s in zip(frames, src):
                    f.write("%s\t%s\n" % (s, "　".join("%s=%s" % (SEATS[j], w[0]) for j, w in enumerate(fr) if w is not None)))
                f.write("\n")
    frames = [f for t in talks for f in t[0]]
    res = dict(話=len(talks), 枠=len(frames), 三つの席だけの枠=sum(is_core(f) for f in frames), 外した数=why,
               席ごとの枠の数={s: sum(f[j] is not None for f in frames) for j, s in enumerate(SEATS)},
               席の数ごとの枠={str(n): sum(sum(w is not None for w in f) == n for f in frames) for n in range(2, 6)},
               語の持ち物=ZP.cover(talks, wave_of, full.of))
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    res.update(run(wave_of, full.of, talks))
    res["秒"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
