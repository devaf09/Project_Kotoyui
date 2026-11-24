"""I49zo: 全部の問いと、形ごとの答えをテキストに出す。続けて、ローカル AI に答えの違和感を見てもらう（持ち主の案、未確定の試し）。

持ち主の言葉:「全部テキストに出して、試しにローカルAI君に違和感をチェックしてもらうのどうかな？」
出す（引数は tools/i49zm_two_channel_run.py と同じ。--out がテキスト（タブ区切り）の出し先。リポジトリの外に置く）:
  1 行 = 1 問。始め方、話の番号、隠した席、だれが・何を・どうした（隠した所は ？）、正解、形ごとの答え。
  形: 関係の表 ／ 属性だけ ／ 属性＋固有の位相（混ぜ 1・1,024 本） ／ 属性＋固有の位相＋振幅 ／ 数えて選ぶ（属性だけ）
見てもらう（python tools/i49zo_dump_answers.py 見る --in 出したテキスト --out 記録.json --text 読む用.txt --model モデル名）:
  --old の形と --new の形で答えが違った問いから、等間隔に --n 問を取る。
  古い答えを入れた組・新しい答えを入れた組・正解の組を、それぞれ「出来事として自然か」1〜5 でローカル AI に付けてもらう（同じ組は 1 回だけ）。
  出る数: 3 つの平均、新しい方が自然だった問い・古い方が自然だった問いの数と z、当たり外れで分けた内訳。
  --minutes を超えたら、そこまでの分で止めて書く。途中の点は記録に残り、次に同じ --out で走らせると続きから。
ローカル AI の点は、持ち主の目を補うだけで、正解ではない。
手で置く物: 取る問いの数、1〜5 の物差し、問いの文面、形の中の値（混ぜ具合、本数、直す速さ、temp）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "intervention"))
sys.path.insert(0, HERE)

SEATS = ("だれが", "何を", "どうした")
ID1 = dict(mix=1.0, n_id=1024)
AMP = dict(phase_lr=0.0, rule="毎回", lr=.01, by_seat=True)
VERS = (("関係の表", "表", None, None), ("属性だけ", "解く", None, None), ("属性＋固有の位相", "解く", ID1, None),
        ("属性＋固有の位相＋振幅", "解く", ID1, AMP), ("数えて選ぶ", "数える", None, None))
HEAD = ["始め方", "話", "席", "だれが", "何を", "どうした", "正解"] + [v[0] for v in VERS]
ASK = ("次の 3 つの語を「だれが・何を・どうした」の組として読みます。出来事として自然かどうかを、"
       "1（とても不自然）〜5（とても自然）の整数 1 つだけで答えてください。\nだれが: %s\n何を: %s\nどうした: %s")


class Rec:
    """答えを順に控えるだけの包み。"""

    def __init__(self, inner):
        self.inner, self.ans = inner, []

    def ask(self, si, tri, cands):
        o = self.inner.ask(si, tri, cands)
        self.ans.append(o["答え"])
        return o

    def learn(self, tri):
        self.inner.learn(tri)

    def tick(self):
        self.inner.tick()


def questions(talks):
    """tools/i49za_fold_run.py の stream と同じ順で、問いを並べる。"""
    out = []
    for ti, (tris, words) in enumerate(talks):
        for tri in tris:
            for si in range(3):
                cands = {w for w in words if w not in {tri[j] for j in range(3) if j != si}} | {tri[si]}
                if len(cands) >= 2:
                    out.append((ti, si, tuple(tri)))
    return out


def dump(wave_of, attr_of, talks, aoz):
    import i49e_relation as RL
    import i49h_tally_run as HR
    import i49k_seat_vec as KV
    import i49w_also_right_run as WR
    import i49za_fold_run as ZR
    from i49zb_mix import Mix
    from i49zj_phase_read import PhaseRead
    from i49zm_two_channel import TwoChannel
    ok = WR.ok_sets([t for tris, _ in talks for t in tris] + [tuple(t) for t in aoz])
    qs, rows = questions(talks), []
    for start, pre in (("白紙から", ()), ("青空を先に", aoz)):
        cols = []
        for name, kind, two, read in VERS:
            table = RL.Relation(wave_of, half=HR.HALF)
            inner = None
            if kind != "表":
                sv = KV.SeatVec(attr_of if two is None else TwoChannel(attr_of, **two).of, weight="数", bind="位相")
                inner = PhaseRead(sv, **read) if read else KV.Unbind(sv, online=True)
            for tri in pre:
                table.learn(tri)
                if inner is not None:
                    inner.sv.hear(tri)
                    if read:
                        inner.hear_seen(tri)
            rec = Rec(table if kind == "表" else inner if kind == "解く" else Mix(table, inner, upper=None))
            ZR.stream(rec, talks, ok)
            assert len(rec.ans) == len(qs)
            cols.append(rec.ans)
            print(start, name, round(sum(a == q[2][q[1]] for a, q in zip(rec.ans, qs)) / max(1, len(qs)), 4), flush=True)
        for k, (ti, si, tri) in enumerate(qs):
            shown = ["？" if j == si else tri[j] for j in range(3)]
            rows.append([start, str(ti), SEATS[si]] + shown + [tri[si]] + [str(c[k]) if c[k] is not None else "" for c in cols])
    return rows


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(HEAD) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [dict(zip(head, line.rstrip("\n").split("\t"))) for line in f if line.strip()]


def filled(row, word):
    return tuple(word if row[s] == "？" else row[s] for s in SEATS)


def pick(rows, start, old, new, n):
    """答えが違った問いから、等間隔に n 問。"""
    diff = [r for r in rows if r["始め方"] == start and r[old] and r[new] and r[old] != r[new]]
    if len(diff) <= n:
        return diff, len(diff)
    return [diff[(i * len(diff)) // n] for i in range(n)], len(diff)


def look(rows, score, start="白紙から", old="属性だけ", new="属性＋固有の位相", n=400, minutes=25, seen=None, save=None):
    """score(組) → 1〜5 か None。seen = 前の点（組の文字列 → 点）。"""
    seen = {} if seen is None else seen
    chosen, total = pick(rows, start, old, new, n)
    t0, done = time.time(), []
    for r in chosen:
        if time.time() - t0 > minutes * 60:
            break
        pts = []
        for w in (r[old], r[new], r["正解"]):
            key = "｜".join(filled(r, w))
            if key not in seen:
                seen[key] = score(filled(r, w))
                if save:
                    save(seen)
            pts.append(seen[key])
        done.append((r, pts))
    return stats(done, old, new, total, len(chosen)), done


def stats(done, old, new, total, chosen):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    def part(items):
        both = [(p[0], p[1]) for _, p in items if p[0] is not None and p[1] is not None]
        up, dn = sum(a < b for a, b in both), sum(a > b for a, b in both)
        return dict(問い=len(items), 古い答えの平均=mean([p[0] for _, p in items]), 新しい答えの平均=mean([p[1] for _, p in items]),
                    正解の平均=mean([p[2] for _, p in items]), 新しい方が自然=up, 古い方が自然=dn, 同じ=len(both) - up - dn,
                    z=round((up - dn) / max(1.0, (up + dn) ** 0.5), 2))
    out = dict(答えが違った問い=total, 取った問い=chosen, 見終わった問い=len(done), 点が読めなかった数=sum(x is None for _, p in done for x in p), 全体=part(done))
    out["新しい方が当たり"] = part([d for d in done if d[0][new] == d[0]["正解"]])
    out["古い方が当たり"] = part([d for d in done if d[0][old] == d[0]["正解"]])
    out["どちらも外れ"] = part([d for d in done if d[0][new] != d[0]["正解"] and d[0][old] != d[0]["正解"]])
    out["席ごと"] = {s: part([d for d in done if d[0]["席"] == s]) for s in SEATS}
    return out


def write_text(path, done, old, new):
    with open(path, "w", encoding="utf-8") as f:
        f.write("問い（？ が隠した所）\t正解\t%s\t点\t%s\t点\t正解の点\n" % (old, new))
        for r, p in done:
            f.write("%s が %s を %s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (r["だれが"], r["何を"], r["どうした"], r["正解"], r[old], p[0], r[new], p[1], p[2]))


def main_look(argv):
    import argparse
    from ollama_client import Ollama
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", default="白紙から")
    ap.add_argument("--old", default="属性だけ")
    ap.add_argument("--new", default="属性＋固有の位相")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--minutes", type=float, default=25)
    a = ap.parse_args(argv)
    ai = Ollama(a.model, system="あなたは日本語の文の自然さを見る係です。数字 1 つだけで答えます。")
    seen = {}
    if os.path.exists(a.out):
        seen = json.load(open(a.out, encoding="utf-8")).get("点", {})

    def score(tri):
        try:
            m = re.search(r"[1-5]", ai._post([dict(role="system", content=ai.system), dict(role="user", content=ASK % tri)], temperature=0.0, max_tokens=8))
        except OSError:
            return None
        return int(m.group()) if m else None

    res = {}

    def save(s):
        if len(s) % 20 == 0:
            json.dump(dict(res, 点=s), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    t0 = time.time()
    st, done = look(read_rows(a.src), score, a.start, a.old, a.new, a.n, a.minutes, seen, save)
    res = dict(モデル=a.model, 始め方=a.start, 古い形=a.old, 新しい形=a.new, 秒=round(time.time() - t0), **st)
    json.dump(dict(res, 点=seen), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_text(a.text, done, a.old, a.new)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "見る":
        return main_look(sys.argv[2:])
    import i49h_tally_run as HR
    import i49m_attr_base as AB
    i = sys.argv.index("--content")
    content = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    a, vec_of, talks, aoz, t0 = HR.load()
    full = AB.AttrBase(json.load(open(content, encoding="utf-8")))
    rows = dump(vec_of, full.of, talks, aoz)
    write_rows(a.out, rows)
    print("行", len(rows), "秒", round(time.time() - t0))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
