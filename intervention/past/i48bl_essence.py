"""I48bl: エッセンスの板（持ち主の案「介入のループが解決した時、次のループを少なくする、こうするといい、みたいな物をたくさん集める」、未確定）。

型は I48bi・bk と同じ（UModel・型 C、5 万文で学ぶ）。学びに使っていない文（参照の文）を 1 文ずつ流し、新しい経験として扱う。
1 文ごとに:
  1. 今の型（本体 ＋ 介入の単位 ＋ エッセンスの板）で選び、間ごとに答えを使わない 6 種類の違和感を出す。
     名札は、本体で試しの文を照らした時の値の並びの中での順位の割合がいちばん大きい種類。迷い・自信のなさ_選び のどちらかが、
     その並びの上位 1 割の線を越えた間を「ループに入った間」とする。
  2. ループに入った間のうち、エッセンスの板に当たる鍵がある間は、エッセンスで方向を押して選び直す（ループを回さずに直す）。
     押した後に線を下回れば「エッセンスで直した」。下回らなければ、または当たる鍵が無ければ「教わる」:
     その文を介入の単位として教え（正解の詰め物を使う）、エッセンスの板に「鍵 → 正解の詰め物」を 1 件足す。
  3. エッセンスを押した間では、押した後の選びが正解なら「効いた」と数える。押しの強さは手で置かず、その鍵の効いた割合
     （(効いた + 1) / (押した + 2)）× log(1 + その鍵でその詰め物を教わった回数) にする。
鍵の粗さ（持ち主: 粗い方を主に）:
  粗い: (名札, 左の語の品詞) ・(名札, 右の語の品詞) ・(名札, 文末か) の 3 つ。1 つの間で 3 つ全部が鳴り、押しは足す（足し算）か、
        平均（そろい具合: 鳴ったエッセンスの押しの平均）で読む
  中  : (名札, 左の語の D の帯の属性, 右の語の D の帯の属性, 文末か)
  細かい: (名札, 左の語, 右の語, 文末か)
  比べ: エッセンス無し（教わった物は介入の単位だけに貯める） ／ 粗い・置き換え（押しでなく、エッセンスのいちばん多い詰め物に置き換える）
測る物（1,000 文ごと）: ループに入った間、教わった間、エッセンスで直した間、エッセンスで直したのに正解から外した間（害）、
  エッセンスで直した所がまた線を越える割合（直したつもりで直っていない）。最後に試しの 1,000 文の 間・文末・文・助詞ごとの害。
  貯まったエッセンスの数、よく使われた上位（鍵と詰め物）、1 回しか使われない物の割合。
融合の下調べ（粗い鍵だけ、数えだけ）: 同じ間で、違う鍵が同じ詰め物を押した 2 つのエッセンスの組の回数を、別々に鳴る割合から見込む回数と比べ、
  見込み ＋ √(見込み × 2 ln V) を越える組だけを結ぶ。エッセンスごとに結ばれた相手の集まりを「まとまり」とし（かぶりあり）、
  まとまりの数・大きさ・2 つ以上のまとまりに入るエッセンスの数。
手で置く物: 流す文の数 5,000、1,000 文ごとの区切り、上位 1 割の線、介入の板を引き直す間隔（100 文）、鍵の作り方（上の 3 通り）。
python intervention/past/i48bl_essence.py --conllu ... --content ... --func ... --out out.json [--train 50000] [--test 1000] [--stream 5000]
"""
from __future__ import annotations
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする

import argparse
import json
import math
import os
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import i48ab_long_grain as G  # noqa: E402
import i48ad_lookahead as AD  # noqa: E402
import i48ae_marks as AE  # noqa: E402
import i48aj_resonance as AJ  # noqa: E402
import i48ak_teach as AK  # noqa: E402
import i48ao_many_particles as AO  # noqa: E402
import i48ap_units as AP  # noqa: E402
import i48at_recheck as RC  # noqa: E402
import i48bi_odd_kinds as BI  # noqa: E402

LOOP_KINDS = ("迷い", "自信のなさ_選び")
CHUNK = 1000
REFRESH = 100


class Essence:
    """エッセンスの板: 鍵 → 詰め物の数え、と鍵ごとの効いた割合。本体にも介入の板にも混ぜない。記録を残し、取り消せる。"""

    def __init__(self, nc):
        self.nc = nc
        self.cnt = defaultdict(Counter)          # 鍵 → 詰め物 → 教わった回数
        self.applied, self.effect = Counter(), Counter()
        self.used = Counter()                    # (鍵, 詰め物) → 押しに使われた回数
        self.log = []

    def add(self, key, c, src):
        self.cnt[key][c] += 1
        self.log.append(dict(番号=len(self.log), 鍵=list(key), 詰め物=int(c), 出どころ=src, 取り消し=False))

    def undo(self, numbers):
        for n in numbers:
            e = self.log[n]
            if not e["取り消し"]:
                e["取り消し"] = True
                self.cnt[tuple(e["鍵"])][e["詰め物"]] -= 1

    def push(self, keys, mode):
        """鍵の並び（1 つの間で鳴る鍵）→ 詰め物ごとの押し（足し算 か 平均）と、鳴ったエッセンス。"""
        vs, fired = [], []
        for k in keys:
            c = self.cnt.get(k)
            if not c:
                continue
            r = (self.effect[k] + 1) / (self.applied[k] + 2)
            v = np.zeros(self.nc, dtype=np.float32)
            for cls, n in c.items():
                if n > 0:
                    v[cls] = r * math.log1p(n)
                    fired.append((k, cls))
            vs.append(v)
        if not vs:
            return None, fired
        return (np.sum(vs, axis=0) if mode != "平均" else np.mean(vs, axis=0)), fired


def keys_of_gap(S, keys, x, label, grain):
    i, last = x["i"], x["last"]
    L = keys[i]
    R = keys[i + 1] if i + 1 < len(keys) else ("文末", "文末")
    if grain == "粗い":
        return [(label, "左", L[1]), (label, "右", R[1]), (label, "末", bool(last))]
    if grain == "中":
        def d(k):
            w = S.cb.get(k[0])
            D = (w or {}).get("attrs", {}).get("D", {}) if w else {}
            return max(D, key=D.get) if D else k[1]
        return [(label, d(L), d(R), bool(last))]
    return [(label, L[0], R[0], bool(last))]


def gap_rows(m, s, nat, push_of=None):
    """BI.rows_of と同じ値を 1 文で出す。push_of(x) が押しの並び（詰め物ごと）を返せば、その間の点に足して選ぶ。"""
    xs = AE.items(m.S, s, "節")
    keys = AK.keys_of(s)
    fic = m.who.fiction(keys)
    ups = {u for _, u in keys}
    placed, rows = [], []
    for x in xs:
        g = m.gap_scores(keys, x, fic)
        r3 = m.gram.score(m.gram.state(ups, placed, x["last"]))
        logic, wave = g["R2"] + r3, g["R1"]
        three = wave + logic
        both = g["今"] + three
        if push_of is not None:
            pv = push_of(x)
            if pv is not None:
                both = both + pv
        three[-1] = both[-1] = -1e9
        cand = (set(np.argsort(-g["今"])[:AK.K]) | set(np.argsort(-three)[:AK.K])) - {m.nc - 1}
        if push_of is not None and push_of(x) is not None:
            cand |= {int(c) for c in np.nonzero(push_of(x) > 0)[0] if c != m.nc - 1}
        pick = max(cand, key=lambda c: both[c])
        top2 = np.sort(both)[-2:]
        p = np.exp(both - both.max())
        p /= p.sum()
        lg, wv = logic.copy(), wave.copy()
        lg[-1] = wv[-1] = -1e9
        pred = AJ.next_pred(keys, x["i"])
        rows.append(dict(i=int(x["i"]), x=x, gold=int(m.cid(x["y"])), pick=int(pick), 迷い=-float(top2[1] - top2[0]), 自信のなさ_選び=-float(p[pick]),
                         筋と波の食い違い=float(AJ.ranks(lg)[pick] + AJ.ranks(wv)[pick]),
                         筋の無い共振=float(lg[pick] <= np.median(lg[:-1]) and int(np.argmax(wv)) == pick),
                         組の珍しさ=-nat.score(m.pg.cls[pick], pred[0] if pred else None),
                         手持ちに無い=float(np.exp(g["今"][-1] - np.logaddexp.reduce(g["今"])))))
        placed.append(m.pg.cls[pick])
    return keys, xs, rows


class EssScore:
    """差し込み: その時の文で、押す間とその押しを持つ（gap_rows の push_of を本体の walk でも効かせるため）。"""

    def __init__(self):
        self.cur = None

    def clone(self):
        return self

    def refresh(self, m):
        pass

    def score(self, m, keys, x, f, v):
        if self.cur is None:
            return np.zeros(m.nc, dtype=np.float32)
        return self.cur(keys, x)


def main():
    ap = argparse.ArgumentParser()
    for k in ("conllu", "content", "func", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--stream", type=int, default=5000)
    ap.add_argument("--D", type=int, default=8192)
    ap.add_argument("--r1", default="架空・前")
    a = ap.parse_args()
    t0 = time.time()
    sents = [s for s in G.read_conllu(a.conllu) if s["toks"]]
    te = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 == 0][: a.test]
    tr_all = [s for s in sents if zlib.crc32(s["id"].encode()) % 20 != 0]
    hold, tr = tr_all[-2000:], tr_all[:-2000][: a.train]
    stream = [s for s in tr_all[:-2000][a.train:] if s["toks"]][: a.stream]
    cb = json.load(open(a.content, encoding="utf-8"))
    func = G.Func(json.load(open(a.func, encoding="utf-8")))
    S = G.AttrSpace(cb, func, a.D, bands=("B", "C", "D", "E", "G"))
    fill, surf = AD.Fill(S), G.Surface()
    for s in tr:
        fill.see(s)
        cont, gaps, _ = G.split(s)
        for (l, f_, u), g in zip(cont, gaps):
            surf.add(l, f_, u, g)
    m = AP.UModel(S, fill, a.r1).fit(tr, hold).init_units("C")
    m.refresh()
    nat = RC.Natural(fill, tr)
    rows_te = BI.rows_of(m, te, nat)
    ref = {kd: np.sort([r[kd] for r in rows_te]) for kd in BI.LABEL_KINDS}
    thr = {kd: float(np.quantile(ref[kd], 0.9)) for kd in LOOP_KINDS}

    def label(r):
        pc = {kd: np.searchsorted(ref[kd], r[kd]) / len(ref[kd]) for kd in BI.LABEL_KINDS}
        return max(pc, key=pc.get)

    def flagged(r):
        return any(r[kd] > thr[kd] for kd in LOOP_KINDS)
    res = dict(学びの文=len(tr), 流した文=len(stream), 試しの文=len(te), 区切り=CHUNK, 線=thr, 型={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def push_for(ess, ks, mode):
        if mode == "置き換え":
            best = None
            for k in ks:
                c = ess.cnt.get(k)
                if c:
                    cls, n = c.most_common(1)[0]
                    if n > 0 and (best is None or n > best[1]):
                        best = (cls, n, k)
            if best is None:
                return None
            v = np.zeros(m.nc, dtype=np.float32)
            v[best[0]] = 1e3
            return v
        return ess.push(ks, mode)[0]

    def test_eval(mm, es, ess, grain, mode):
        """試しの文: 名札の立った間にエッセンスを押して選び直した時の、間・文末・文の当たりと、正解から外した数（害）、助詞ごとの当たり。"""
        hit = Counter()
        tot = Counter()
        harm = changed = 0
        per = defaultdict(Counter)
        for s in te:
            if not AE.items(S, s, "節"):
                continue
            es.cur = None
            keys, xs, rows = gap_rows(mm, s, nat)
            final = {r["i"]: r for r in rows}
            if grain is not None:
                pushes = {}
                for r in rows:
                    if flagged(r):
                        v = push_for(ess, keys_of_gap(S, keys, xs[r["i"]], label(r), grain), mode)
                        if v is not None:
                            pushes[r["i"]] = v
                if pushes:
                    es.cur = lambda ks_, x_: pushes.get(x_["i"], np.zeros(m.nc, dtype=np.float32))
                    _, _, rows2 = gap_rows(mm, s, nat)
                    es.cur = None
                    for r0, r1 in zip(rows, rows2):
                        if r1["pick"] != r0["pick"]:
                            changed += 1
                            harm += r0["pick"] == r0["gold"] and r1["pick"] != r1["gold"]
                    final = {r["i"]: r for r in rows2}
            ok_all = True
            for r in final.values():
                w = "文末" if r["x"]["last"] else "間"
                tot[w] += 1
                ok = r["pick"] == r["gold"]
                hit[w] += ok
                ok_all &= ok
                y = r["x"]["y"]
                if y in ("と", "で", "に", "を", "が", "は", "も", "へ", "から"):
                    per[y]["数"] += 1
                    per[y]["当たり"] += ok
            tot["文"] += 1
            hit["文"] += ok_all
        es.cur = None
        return dict(**{w: round(hit[w] / max(1, tot[w]), 4) for w in ("間", "文末", "文")}, 替えた間=changed, 正解から外した間=harm,
                    助詞={y: round(c["当たり"] / max(1, c["数"]), 3) for y, c in per.items()})
    configs = [("エッセンス無し", None, None), ("粗い・押し・足し算", "粗い", "足し算"), ("粗い・押し・平均", "粗い", "平均"),
               ("粗い・置き換え", "粗い", "置き換え"), ("中・押し", "中", "足し算"), ("細かい・押し", "細かい", "足し算")]
    if os.environ.get("BL_CONFIGS"):                       # 回す形を絞る（名前の , 区切り）
        keep = os.environ["BL_CONFIGS"].split(",")
        configs = [c for c in configs if c[0] in keep]
    for nm, grain, mode in configs:
        t1 = time.time()
        mm = m.clone()
        mm.kind = "C"
        ess = Essence(m.nc)
        es = EssScore()
        mm.extras.append(es)
        mm.refresh()
        curve, cur = [], Counter()
        cofire = Counter()
        fire_n = Counter()
        n_gaps = 0
        pending = 0
        for si, s in enumerate(stream):
            if not AE.items(S, s, "節"):
                continue
            es.cur = None
            keys, xs, rows = gap_rows(mm, s, nat)
            flags = {r["i"]: label(r) for r in rows if flagged(r)}
            cur["間"] += len(rows)
            n_gaps += len(rows)
            if not flags:
                pass
            else:
                cur["ループに入った間"] += len(flags)
                before = {r["i"]: r for r in rows}
                taught = False
                pushes = {}
                if grain is not None:
                    for i, lab in flags.items():
                        ks = keys_of_gap(S, keys, xs[i], lab, grain)
                        if mode == "置き換え":
                            best = None
                            for k in ks:
                                c = ess.cnt.get(k)
                                if c:
                                    cls, n = c.most_common(1)[0]
                                    if n > 0 and (best is None or n > best[1]):
                                        best = (cls, n, k)
                            if best is not None:
                                v = np.zeros(m.nc, dtype=np.float32)
                                v[best[0]] = 1e3
                                pushes[i] = (v, [(best[2], best[0])])
                        else:
                            v, fired = ess.push(ks, mode)
                            if v is not None:
                                pushes[i] = (v, fired)
                        if grain == "粗い":
                            ids = sorted({(k, cls) for k in ks for cls, n in ess.cnt.get(k, {}).items() if n > 0})
                            for e in ids:
                                fire_n[e] += 1
                            for p in range(len(ids)):
                                for q in range(p + 1, len(ids)):
                                    if ids[p][0] != ids[q][0] and ids[p][1] == ids[q][1]:   # 違う鍵が同じ詰め物を押す時だけ（同じ鍵の別の詰め物は数えない）
                                        cofire[(ids[p], ids[q])] += 1
                after = {}
                if pushes:
                    es.cur = lambda ks_, x_: pushes[x_["i"]][0] if x_["i"] in pushes else np.zeros(m.nc, dtype=np.float32)
                    _, _, rows2 = gap_rows(mm, s, nat)
                    es.cur = None
                    after = {r["i"]: r for r in rows2}
                for i, lab in flags.items():
                    r0 = before[i]
                    if i in pushes:
                        r1 = after[i]
                        cur["エッセンスを押した間"] += 1
                        ks_fired = {k for k, _ in pushes[i][1]}
                        for k in ks_fired:
                            ess.applied[k] += 1
                            ess.effect[k] += r1["pick"] == r1["gold"]
                        for e in pushes[i][1]:
                            if e[1] == r1["pick"]:                  # 選ばれた詰め物を押したエッセンスだけを「使われた」に数える
                                ess.used[e] += 1
                        if r0["pick"] == r0["gold"] and r1["pick"] != r1["gold"]:
                            cur["エッセンスの害"] += 1
                        if not flagged(r1):
                            cur["エッセンスで直した"] += 1
                            cur["エッセンスで直して正解"] += r1["pick"] == r1["gold"]
                            continue
                        cur["押しても線を越えたまま"] += 1
                    # 教わる（正解の詰め物で、その文を単位として）
                    cur["教わった間"] += 1
                    if grain is not None:
                        for k in keys_of_gap(S, keys, xs[i], lab, grain):
                            ess.add(k, r0["gold"], si)
                    taught = True
                if taught:
                    mm.teach_unit(("教わる", nm, keys, G.split(s)[1]), xs=xs)
                    pending += 1
                    if pending >= REFRESH:
                        mm.refresh()
                        pending = 0
            if (si + 1) % CHUNK == 0:
                curve.append(dict(文=si + 1, **{k: int(v) for k, v in cur.items()}))
                cur = Counter()
                print(nm, json.dumps(curve[-1], ensure_ascii=False), flush=True)
        mm.refresh()
        es.cur = None
        r = AO.measure(mm, te, surf)                       # 介入の単位だけ（エッセンスを押さない）
        te_res = test_eval(mm, es, ess, grain, mode)       # 試しの文で、名札の立った間にエッセンスを押した時
        out = dict(曲線=curve, 試し_単位だけ=dict(間=r["間"], 文末=r["文末"], 文=r["文"]), 試し_エッセンスも=te_res, 秒=round(time.time() - t1),
                   介入の単位=len([u for u in mm.units if not u["取り消し"]]))
        if grain is not None:
            n_ess = sum(1 for k, c in ess.cnt.items() for cls, n in c.items() if n > 0)
            used = Counter({e: n for e, n in ess.used.items()})
            out["エッセンス"] = dict(数=n_ess, 押しに使われた物=len(used), 一回だけ使われた割合=round(sum(1 for n in used.values() if n == 1) / max(1, len(used)), 4),
                                 よく使われた上位=[dict(鍵=[str(x) for x in e[0]], 詰め物=m.pg.cls[e[1]], 回数=n) for e, n in used.most_common(15)],
                                 効いた割合_全体=round(sum(ess.effect.values()) / max(1, sum(ess.applied.values())), 4))
        if grain == "粗い" and fire_n:
            V = max(2, len(fire_n))
            tot = max(1, n_gaps)
            links = defaultdict(set)
            n_pairs = 0
            for (e1, e2), n in cofire.items():
                exp = fire_n[e1] * fire_n[e2] / tot
                if n > exp + math.sqrt(exp * 2 * math.log(V)):
                    links[e1].add(e2)
                    links[e2].add(e1)
                    n_pairs += 1
            groups = {frozenset([e] + sorted(links[e])) for e in links}
            member = Counter(e for g in groups for e in g)
            out["融合の下調べ"] = dict(鳴ったエッセンス=len(fire_n), 結んだ組=n_pairs, まとまり=len(groups),
                                  まとまりの大きさ=dict(Counter(min(len(g), 10) for g in groups)),
                                  二つ以上のまとまりに入るエッセンス=sum(1 for v in member.values() if v >= 2))
        res["型"][nm] = out
        print(nm, json.dumps({k: v for k, v in out.items() if k != "曲線"}, ensure_ascii=False)[:1500], flush=True)
        save()
        del mm, ess
    res["秒"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
