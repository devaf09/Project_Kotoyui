"""I49g: 足し算 A＋B＝C を、関係の表（intervention/i49e_relation.Relation、変えない）に聞かせて、覚える途中の反応を数える。数えだけ（持ち主の案、未確定）。

持ち主の案: もっとシンプルな仕組みで試す。A＋B＝C を 1 桁の数で行い、C が出るまでの反応の変化を学習と共に見る。それを 2 桁の足し算まで成長できるか。
数は文字の並びとして扱う（"7"、"12"）。3 つ組は (A, B, C)。C を隠して問う。候補は "0"〜"18"（2 桁では "0"〜"198"）。

1. 1 桁の 100 問を、決まった順で 1 問ずつ: 問う → 答え合わせ → 覚える。周ごとに、当たり・答えた段・外れの大きさ（|答え − C| の平均）・白紙。
2. 見せていない問い: 100 問のうち 20 問を取り分け、残り 80 問を覚えた後に 20 問を問う。
3. 2 桁（10〜99 どうし、8,100 問）: 1 桁を覚えた表に、(ア) 数をそのまま 1 語として問う、(イ) 桁に分けて、1 の位から順に 1 桁の問いにして問う
   （分け方と繰り上がりの運び方は、道具が手で与える。表が持つのは 1 桁の事実だけ）。1 桁を何問覚えた所で、2 桁が何問解けるか。
4. 思考の波: 1 と同じ流れを、感情・状態の器（I49a）と違和感の波（I49d の OddWave）を通して回す。1 問 = A を聞く → B を聞く → 表の答えを自分で言う
   （settle。白紙なら 1 刻み待つ）→ 正解の C を聞く → 答え合わせの違和感（答えが違った・白紙）を波へ → 覚える。
   器の波の大きさ・感度・C を聞いた時の外れ、違和感の波を、最初の問い・1 周目の終わり・最後の問いで丸ごと残す（--waves で npz にも）。周ごとの平均も出す。
   器に聞かせる向きは、語の向きが なし の時は数ごとに別々の向き（どの数どうしも似ていない）。
5. 答える前の違和感（正解は使わない）: 借りた = 聞いた事のある組（2 語の段）でなく、下の段で答えたら 1（白紙も 1）。迷い = 2 番目の点 ÷ 1 番目の点（候補が 1 つなら 0）。
   80 問を覚えた後、見せた 80 問と見せていない 20 問で、この 2 つの平均を並べる。線は引かない。
6. ずれを運ぶ（intervention/i49g_shift.Shifter）: 数え歌 "0 1 … 18" を先に聞いて並びを覚え、事実を聞くたびにずれの決まりを数える。語の向きは使わない。
   (ア) 80 問を覚えた後の、見せていない 20 問。(イ) 白紙から 100 問を 1 問ずつ「問う → 答え合わせ → 覚える」時の、25 問ごとの当たりと段。
   (ウ) 数え歌を聞かせない時（並びを知らない）。
   (エ) 同じ部品を、引き算（A ≧ B の 55 問）と掛け算（100 問、数え歌は 0〜81）に当てる。決まりが足し算用に置かれた物でなく、数えた物かを見る。
7. 新しい数が 1 回だけ出る（持ち主の案）: 0〜8 だけの世界（数え歌 0〜8、足して 8 までの 45 問）を覚えた後、9 を 1 回だけ見せ、
   9 を使う見せていない問い（足して 9 になる組、見せた 1 問を除く）が当たるかを数える。候補は、聞いた事のある数。
   出方: 出る前 ／ 並びとして（「8 の次は 9」）／ 答えとして 1 回（4＋5＝9）で並びを埋めない ／ 同じで並びを埋める（Shifter(fill=True)）。
   続けて 10、11、… 16 も、それぞれ答えとして 1 回だけ見せて、同じ数えをくり返す。
8. 50 までを覚えて、50 より上を計算する（持ち主の案）: 足して 50 以下の全部（1,326 問、1 桁どうしも入る）を覚え、足して 51〜99 の問い（7 問に 1 問）を問う。
   数は 1 語のまま（桁に分けない）。答えは「数え足し」（聞いた事のある事実に着くまで 1 歩ずつ戻り、その分だけ答えを運ぶ）で出す。
   (ア) 数え歌を 0〜99 まで聞いている。(イ) 数え歌は 0〜50 まで（51 より上の数の名前を知らない）。
   (ウ) 数え歌は 0〜50 まで。問いを答えの小さい順に 1 問ずつ「問う → 正解を聞く → 覚える」。並びを自分で埋める。初めて出る数は必ず外れるので、それを除いた当たりも出す。
   (エ) (ウ) を、決まった飛ばし方で混ぜた順で。
9. 数え歌を文字で見る（intervention/i49g_digits.CharSong。桁を教えずに見つける案 A）: 数え歌を 0〜n まで聞いた後、n より上 98 までの数の「次」を自分で作れるか。
   n = 9、10、19、20、30、50。作れた数、間違えた例。
   8 の (オ): 数え歌は 0〜50 まで。文字の決まりで 51 より上の並びを自分で作ってから、50 より上の足し算を問う。
語の向き: なし ／ 数え歌（"0 1 2 … 18" を聞いた時の、隣どうしの出方を数えた向き。n の向き = n−1・n・n+1 の所が 1）。
乱数は使わない。手で置く物: 問いの順（37 飛ばし）、取り分ける 20 問（(A + 2B) を 5 で割り切れる組）、数え歌の窓（隣 1 つ）、周の数。
python tools/i49g_add_count.py --out <stats.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
from i49e_relation import Relation  # noqa: E402

TOP = 19


def song_vec(w):
    """数え歌の向き: n−1・n・n+1 の所が 1。数え歌に出ない語は None。"""
    if not w.isdigit() or int(w) >= TOP:
        return None
    x = np.zeros(TOP, dtype=np.complex64)
    for k in (int(w) - 1, int(w), int(w) + 1):
        if 0 <= k < TOP:
            x[k] = 1.0
    return x


VECS = {"なし": lambda w: None, "数え歌": song_vec}
CANDS = [str(i) for i in range(TOP)]


def facts():
    order = [(i * 37) % 100 for i in range(100)]
    return [(str(i // 10), str(i % 10), str(i // 10 + i % 10)) for i in order]


def held(t):
    return (int(t[0]) + 2 * int(t[1])) % 5 == 0


def ask_c(rel, a, b, cands=CANDS):
    return rel.ask(2, (a, b, None), cands)


def count(rows):
    n = max(1, len(rows))
    ans = [r for r in rows if r["答え"] is not None]
    return dict(問い=len(rows), 当たり=round(sum(r["当たり"] for r in rows) / n, 4),
                白紙=round(1 - len(ans) / n, 4),
                外れの大きさ=round(float(np.mean([abs(int(r["答え"]) - int(r["正解"])) for r in ans])), 3) if ans else None,
                段={k: round(v / n, 4) for k, v in Counter(r["段"] for r in rows).items()})


def one(rel, t):
    o = ask_c(rel, t[0], t[1])
    return dict(正解=t[2], 答え=o["答え"], 段=o["段"], 当たり=o["答え"] == t[2])


def stream(vec_of, rounds=4, block=25):
    """1. 問う → 答え合わせ → 覚える。周ごとと、1 周目は block 問ごと。"""
    rel, out, first = Relation(vec_of), [], []
    for r in range(rounds):
        rows = []
        for t in facts():
            rows.append(one(rel, t))
            rel.learn(t)
            rel.tick()
        out.append(count(rows))
        if r == 0:
            first = [count(rows[i:i + block]) for i in range(0, 100, block)]
    return dict(周ごと=out, 一周目=first)


def holdout(vec_of, rounds=3):
    """2. 80 問を覚えた後、見せていない 20 問を問う。"""
    rel = Relation(vec_of)
    for _ in range(rounds):
        for t in facts():
            if not held(t):
                rel.learn(t)
                rel.tick()
    return count([one(rel, t) for t in facts() if held(t)])


def add_by_digits(rel, x, y):
    """(イ) 桁に分けて 1 の位から。分け方と繰り上がりの運び方は手で与える。表に無い事実に当たったら None。"""
    def fact(a, b):
        o = ask_c(rel, a, b)
        return o["答え"] if o["段"] == "2語" else None
    xs, ys, carry, digits = x[::-1], y[::-1], "0", []
    for i in range(max(len(xs), len(ys))):
        s = fact(xs[i] if i < len(xs) else "0", ys[i] if i < len(ys) else "0")
        if s is None:
            return None
        c2 = "1" if len(s) == 2 else "0"
        d = s[-1]
        if carry == "1":
            s2 = fact(d, "1")
            if s2 is None:
                return None
            if len(s2) == 2:
                c2 = "1"
            d = s2[-1]
        digits.append(d)
        carry = c2
    if carry == "1":
        digits.append("1")
    return "".join(digits[::-1])


def two_digit(vec_of, marks=(25, 50, 75, 100)):
    """3. 1 桁を n 問覚えた所で、2 桁の 8,100 問のうち何問が当たるか。"""
    rel, out, n = Relation(vec_of), {}, 0
    probs = [(str(a), str(b), str(a + b)) for a in range(10, 100) for b in range(10, 100)]
    cands = [str(i) for i in range(199)]
    for t in facts():
        rel.learn(t)
        rel.tick()
        n += 1
        if n in marks:
            whole = Counter()
            hit_w = hit_d = 0
            for a, b, c in probs[::9]:                    # 9 問に 1 問（900 問）
                o = ask_c(rel, a, b, cands)
                whole[o["段"]] += 1
                hit_w += o["答え"] == c
                hit_d += add_by_digits(rel, a, b) == c
            m = len(probs[::9])
            out[str(n)] = dict(問い=m, そのまま_当たり=round(hit_w / m, 4), そのまま_段={k: round(v / m, 4) for k, v in whole.items()},
                               桁に分ける_当たり=round(hit_d / m, 4))
    return out


def doubt(o):
    """答える前の違和感。正解は見ない。"""
    pts = sorted((v for v in o["点"].values() if v > 0), reverse=True)
    return dict(借りた=float(o["段"] != "2語"), 迷い=float(pts[1] / pts[0]) if len(pts) > 1 else 0.0)


def doubt_count(vec_of, rounds=3):
    """5. 見せた問いと見せていない問いで、答える前の違和感を並べる。"""
    rel = Relation(vec_of)
    for _ in range(rounds):
        for t in facts():
            if not held(t):
                rel.learn(t)
                rel.tick()
    out = {}
    for name, flag in (("見せた", False), ("見せていない", True)):
        ds = [doubt(ask_c(rel, t[0], t[1])) for t in facts() if held(t) == flag]
        out[name] = dict(問い=len(ds), 借りた=round(float(np.mean([d["借りた"] for d in ds])), 4),
                         迷い_平均=round(float(np.mean([d["迷い"] for d in ds])), 4),
                         迷い_最小=round(float(min(d["迷い"] for d in ds)), 4), 迷い_最大=round(float(max(d["迷い"] for d in ds)), 4))
    return out


def shift_count(song=True):
    """6. ずれを運ぶ部品。"""
    from i49g_shift import Shifter

    def new():
        sh = Shifter(Relation(lambda w: None))
        if song:
            sh.hear_song(CANDS)
        return sh

    def row(sh, t):
        o = sh.ask(t[0], t[1], CANDS)
        return dict(正解=t[2], 答え=o["答え"], 段=o["段"], 当たり=o["答え"] == t[2])

    sh = new()
    for t in facts():
        if not held(t):
            sh.learn(t)
    out = dict(見せていない問い=count([row(sh, t) for t in facts() if held(t)]),
               決まり={f"{('左', '右')[si]}が{d}": dict(c) for (si, d), c in sorted(sh.rule.items())})
    sh, rows = new(), []
    for t in facts():
        rows.append(row(sh, t))
        sh.learn(t)
    out["一周目"] = [count(rows[i:i + 25]) for i in range(0, 100, 25)]
    out["一周目の全部"] = count(rows)
    return out


def other_ops():
    """6 (エ) 引き算と掛け算。見せていない問い（取り分け方は同じ）の当たりと、数えた決まり。"""
    from i49g_shift import Shifter
    out = {}
    for name, op, ok, top in (("引き算", lambda a, b: a - b, lambda a, b: a >= b, 19), ("掛け算", lambda a, b: a * b, lambda a, b: True, 82)):
        song = [str(i) for i in range(top)]
        fs = [(a, b, str(op(int(a), int(b)))) for a, b, _ in facts() if ok(int(a), int(b))]
        sh = Shifter(Relation(lambda w: None))
        sh.hear_song(song)
        for t in fs:
            if not held(t):
                sh.learn(t)
        te = [t for t in fs if held(t)]
        out[name] = dict(見せていない問い=len(te), 当たり=round(sum(sh.ask(t[0], t[1], song)["答え"] == t[2] for t in te) / max(1, len(te)), 4),
                         決まり={f"{('左', '右')[si]}が{d}": dict(c) for (si, d), c in sorted(sh.rule.items())})
    return out


def new_number(top=8, upto=16):
    """7. 新しい数が 1 回だけ出た後、その数を使う見せていない問いが当たるか。"""
    from i49g_shift import Shifter, STEP

    def world(fill):
        sh = Shifter(Relation(lambda w: None), fill=fill)
        sh.hear_song([str(i) for i in range(top + 1)])
        for a in range(top + 1):
            for b in range(top + 1 - a):
                sh.learn((str(a), str(b), str(a + b)))
        return sh

    def known(sh):
        return sorted({w for si in (0, 2) for w in sh.rel.by[si]} | {t[1] for t in sh.rel.tri if t[1] != STEP}, key=lambda w: (len(w), w))

    def test(sh, n, shown=None):
        ps = [(str(a), str(n - a)) for a in range(n + 1) if (str(a), str(n - a)) != shown]
        cands = known(sh)
        hit = sum(sh.ask(a, b, cands)["答え"] == str(n) for a, b in ps)
        return dict(問い=len(ps), 当たり=hit)

    out = {}
    out["出る前"] = test(world(False), top + 1)
    sh = world(False)
    sh.rel.learn((str(top), STEP, str(top + 1)))
    sh._mv.clear()
    out["並びとして 1 回"] = test(sh, top + 1)
    for name, fill in (("答えとして 1 回・埋めない", False), ("答えとして 1 回・埋める", True)):
        sh, rows = world(fill), {}
        for n in range(top + 1, upto + 1):
            shown = (str(n - 5), "5")                      # 見せる 1 問: (n−5)＋5＝n
            sh.learn(shown + (str(n),))
            rows[str(n)] = test(sh, n, shown)
        out[name] = dict(数ごと=rows, 埋めた並び=list(sh.filled))
    return out


def beyond(limit=50, top=99, every=7):
    """8. limit までを覚えて、その上を数え足しで計算する。"""
    from i49g_shift import Shifter

    def world(song_top, fill):
        sh = Shifter(Relation(lambda w: None), fill=fill)
        sh.hear_song([str(i) for i in range(song_top + 1)])
        for a in range(limit + 1):
            for b in range(limit + 1 - a):
                sh.learn((str(a), str(b), str(a + b)))
        return sh

    probs = [(str(a), str(n - a), str(n)) for n in range(limit + 1, top + 1) for a in range(n + 1)][::every]
    names = [str(i) for i in range(top + 1)]
    out = dict(覚えた事実=(limit + 1) * (limit + 2) // 2, 問い=len(probs))
    for name, song_top in ((f"数え歌 0〜{top}", top), (f"数え歌 0〜{limit}", limit)):
        sh = world(song_top, False)
        cands = names[: song_top + 1]
        rows = [sh.ask(a, b, cands) for a, b, _ in probs]
        hit = [o["答え"] == c for o, (_, _, c) in zip(rows, probs)]
        steps = [o.get("歩", 0) for o in rows]
        out[name] = dict(当たり=round(sum(hit) / len(probs), 4), 段=dict(Counter(o["段"] for o in rows)),
                         歩の平均=round(float(np.mean(steps)), 2), 歩の最大=int(max(steps)))
    from i49g_digits import CharSong
    sh, cs = world(limit, False), CharSong()
    cs.hear(names[: limit + 1])
    made = [str(limit)]
    while len(made) <= top - limit:
        y = cs.next(made[-1])
        if y is None:
            break
        made.append(y)
    sh.hear_song(made)
    cands = names[: limit + 1] + made[1:]
    rows = [sh.ask(a, b, cands) for a, b, _ in probs]
    out[f"数え歌 0〜{limit}・先の並びを文字の決まりで自分で作る"] = dict(
        作った数=len(made) - 1, 正しく作れた数=sum(1 for i, w in enumerate(made[1:]) if w == str(limit + 1 + i)),
        当たり=round(sum(o["答え"] == c for o, (_, _, c) in zip(rows, probs)) / len(probs), 4))
    for name, order in (("答えの小さい順", probs), ("混ぜた順", [probs[(i * 37) % len(probs)] for i in range(len(probs))] if len(probs) % 37 else probs)):
        sh = world(limit, True)
        heard = set(names[: limit + 1])
        rows = []
        for a, b, c in order:
            o = sh.ask(a, b, sorted(heard, key=lambda w: (len(w), w)))
            rows.append((o["答え"] == c, c in heard))
            sh.learn((a, b, c))
            heard.add(c)
        old = [h for h, k in rows if k]
        q = max(1, len(rows) // 4)
        out[f"聞きながら・{name}"] = dict(当たり=round(sum(h for h, _ in rows) / len(rows), 4),
                                    初めて出る数だった問い=sum(1 for _, k in rows if not k),
                                    もう出た数の問いの当たり=round(sum(old) / max(1, len(old)), 4),
                                    四つに分けた当たり=[round(sum(h for h, _ in rows[i * q:(i + 1) * q]) / q, 4) for i in range(4)],
                                    埋めた並びの数=len(sh.filled))
    return out


def char_song(tops=(9, 10, 19, 20, 30, 50), upto=99):
    """9. 数え歌を 0〜n まで聞いた後、n〜98 の数の次を自分で作れるか。"""
    from i49g_digits import CharSong
    out = {}
    for n in tops:
        cs = CharSong()
        cs.hear([str(i) for i in range(n + 1)])
        got = [(m, cs.next(str(m))) for m in range(n, upto)]
        bad = [(str(m), y) for m, y in got if y != str(m + 1)]
        out[str(n)] = dict(問い=len(got), 正しく作れた=len(got) - len(bad), 作れなかった=sum(1 for _, y in bad if y is None),
                           間違えた=sum(1 for _, y in bad if y is not None), 例=bad[:4], 九十九の次=cs.next(str(upto)))
    return out


def one_hot(w):
    x = np.zeros(TOP, dtype=np.complex64)
    x[int(w)] = 1.0
    return x


def waves(vec_of, rounds=4, bands=20):
    """4. 器と違和感の波を通して回す。返す: (数えの dict, npz 用の配列)。"""
    from i49a_inner_vessel import Vessel
    from i49d_two_loops import OddWave
    hear_of = (lambda w: vec_of(w) if vec_of(w) is not None else one_hot(w))
    rel, v = Relation(vec_of), Vessel(TOP, bands=bands)
    odd = OddWave(v.a)
    snaps, per, sur_all = {}, [], []

    def snap(name):
        snaps[name] = dict(大きさ=v.act().copy(), 感度=v.gain.copy(), 外れ=v.last.copy(), 違和感=odd.S.copy(), 種類=list(odd.kinds))

    for r in range(rounds):
        sur, wrong, blank = [], 0, 0
        for i, t in enumerate(facts()):
            v.hear(hear_of(t[0]))
            v.hear(hear_of(t[1]))
            o = ask_c(rel, t[0], t[1])
            if o["答え"] is None:
                v.tick()
            else:
                v.settle(hear_of(o["答え"]))
            v.hear(hear_of(t[2]))                          # 正解を聞く。v.last = 聞く前の状態での外れ（波ごと）
            sur.append(v.last.copy())
            bad, emp = float(o["答え"] is not None and o["答え"] != t[2]), float(o["答え"] is None)
            wrong += bad
            blank += emp
            odd.feel({"答えが違った": bad, "白紙": emp})
            rel.learn(t)
            rel.tick()
            if r == 0 and i == 0:
                snap("最初の問い")
        if r == 0:
            snap("1 周目の終わり")
        m = np.mean(sur, axis=0)
        sur_all.append(m)
        mid = bands // 2
        per.append(dict(違った=wrong / 100, 白紙=blank / 100,
                        正解を聞いた時の外れ=dict(速い波=round(float(m[0]), 4), まん中=round(float(m[mid]), 4), 遅い波=round(float(m[-1]), 4)),
                        感度=dict(速い波=round(float(v.gain[0]), 3), まん中=round(float(v.gain[mid]), 3), 遅い波=round(float(v.gain[-1]), 3)),
                        違和感の波_答えが違った=dict(速い波=round(float(odd.S[0, odd.kinds.index("答えが違った")]), 4),
                                               まん中=round(float(odd.S[mid, odd.kinds.index("答えが違った")]), 4),
                                               遅い波=round(float(odd.S[-1, odd.kinds.index("答えが違った")]), 4))))
    snap("最後の問い")
    out = dict(周ごと=per, 張り付き=int(v.watch["張り付き"].sum()), 死んだ=int(v.watch["死んだ"]))
    arr = {}
    for name, d in snaps.items():
        for k in ("大きさ", "感度", "外れ", "違和感"):
            arr[f"{name}・{k}"] = np.asarray(d[k], dtype=np.float32)
    arr["周ごとの外れ"] = np.stack(sur_all).astype(np.float32)
    return out, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--waves", default=None, help="思考の波の記録（npz）の出力先の頭。語の向きの名前を後ろに付ける")
    a = ap.parse_args()
    res = {}
    for name, vec_of in VECS.items():
        w, arr = waves(vec_of)
        res[name] = dict(一桁=stream(vec_of), 見せていない問い=holdout(vec_of), 二桁=two_digit(vec_of), 思考の波=w, 答える前の違和感=doubt_count(vec_of))
        if a.waves:
            np.savez_compressed(f"{a.waves}_{name}.npz", **arr)
    res["ずれを運ぶ"] = dict(数え歌を聞いた=shift_count(True), 数え歌を聞いていない=shift_count(False), ほかの計算=other_ops(), 新しい数=new_number(), 五十より上=beyond(), 文字の数え歌=char_song())
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
