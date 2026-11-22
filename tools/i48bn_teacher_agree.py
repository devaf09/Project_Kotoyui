"""I48bn: 先生の判定の揺れを、もう記録のある行で数える（持ち主の案「他のモデルを並列で動かして同時に確認するのもあり」、未確定）。

会話を新しく回さない。chat_partner の会話の記録（md）から、往復ごとの「会話の流れ（その時の直前 6 行）＋ボットの行」と先生の段を取り出し、
同じ行を同じ指示文（chat_partner の TEACHER ＋ 判定できないの足し（--unsure）＋ 段の文）で、モデルと温度を替えて何回か判定させる。
1 回ごとの答え（本文のまま）を全部記録する（--raw、リポジトリの外）。数えるのは、答えの JSON の
  判定（判定できない ／ 指摘あり ／ 指摘なし ／ 読めない ／ 考えで切れた（本文が空で length）／ 長さで切れた（本文が途中で length）／ 呼べない ／ 形が違う（has_error の欄が無い））と種類（指摘ありの時だけ）。chat_partner の捨てる決まり（返事に無い所 など）は通さない。
数える物:
  同じモデル・同じ温度の中で: 判定が割れた行（あり・なしが割れた ／ 種類だけ割れた）の数と割合、割れた行の行の作りの内訳
  モデルの間で（各モデルの温度 0.2 のいちばん多い答え）: あり・なしが合う割合、種類が合う割合（両方ありの時）、割れた行と、同じモデルの中で割れた行の重なり
  ボット自身の違和感: 記録の行の作りの説明にある 点・行づくり・作る途中で違和感を出したか・P(指摘) の最大 を、揺れた行と揺れなかった行で比べる（平均と AUC）
揺れた行（同じモデルの中で割れた、またはモデル間で割れた）は --unstable（jsonl、リポジトリの外）に集める。
手で置く物: 考え（think）は記録の回と同じ off、回数 3、温度 0.2・0.7、行の作りの名前は説明の初めの 20 字（「（」より前）、モデル間の比べは温度 0.2 のいちばん多い答え。
python tools/i48bn_teacher_agree.py --log <会話の記録の md> --host http://127.0.0.1:<port> --runs "<モデル>@0.2x3,<モデル>@0.7x3" \
    --raw <raw.jsonl> --unstable <unstable.jsonl> --out <stats.json> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
import chat_partner as CP  # noqa: E402
import openai_chat as OC  # noqa: E402


NOT_ANSWER = ("読めない", "考えで切れた", "長さで切れた", "呼べない", "形が違う")   # 答えとして数えない物（割れ方・いちばん多い答えから除く）


def read_turns(path):
    """記録の md → 往復ごとの dict（会話・往復・段・流れ・行・行の作り・記録の先生の答え）。"""
    turns, hist, conv, ti, cur = [], [], 0, 0, None
    for ln in open(path, encoding="utf-8").read().split("\n"):
        if ln.startswith("## 会話 "):
            conv, ti, hist, cur = int(ln.split()[-1]), 0, [], None
            continue
        m = re.match(r"- 相手: (.*)$", ln)
        if m:
            hist.append(f"相手: {m.group(1)}")
            continue
        m = re.match(r"- ボット: \*\*(.*)\*\* （(.*)）$", ln)
        if m:
            ti += 1
            line, why = m.group(1), m.group(2)
            cur = dict(会話=conv, 往復=ti, 流れ=hist[-6:], 行=line, 行の作り=re.split(r"[「:：（]", why)[0].strip()[:12] or "（なし）", 説明=why, 段=None, 記録の答え=None)
            turns.append(cur)
            hist.append(f"ボット: {line}")
            continue
        m = re.match(r"\s+- 先生（段 (\d)）(?:［(.*?)］|: (見立てを捨てた))", ln)
        if m and cur is not None and cur["段"] is None:
            cur["段"] = int(m.group(1))
            cur["記録の答え"] = m.group(2) or m.group(3)
    return turns


def own_odd(why):
    """ボット自身の値（説明の文から）: 点・行づくり（最後の物）、作る途中の違和感、P(指摘) の最大。"""
    pt = re.findall(r"点 (-?[0-9.]+)", why)
    gz = re.findall(r"行づくり (-?[0-9.]+)", why)
    pp = [float(x) for x in re.findall(r"P\(指摘\) ([0-9.]+)", why)]
    return dict(点=float(pt[-1]) if pt else None, 行づくり=float(gz[-1]) if gz else None,
                作る途中の違和感=float("違和感" in why), P指摘の最大=max(pp) if pp else 0.0)


def judge(host, model, t, unsure, temperature, think=False, api=None):
    """api = None なら ollama（host）。api = dict(url, key_env, budget, reasoning_effort, chat_template_kwargs, max_tokens) なら OpenAI 互換の口。"""
    hist = "\n".join(t["流れ"])
    extra = (CP.UNSURE[unsure] + CP.UNSURE_FIELDS) if unsure != "off" else ""
    msgs = [dict(role="system", content=CP.TEACHER + extra + CP.STAGES[t["段"] or 1]),
            dict(role="user", content=f"会話の流れ:\n{hist}\n\nボットの返事:\n{t['行']}")]
    t0 = time.time()
    meta = {}
    if api is None:
        text = CP.ollama_chat(host, model, msgs, temperature=temperature, first_line=False, num_predict=2048, fmt="json", think=think)
    else:
        o = OC.chat(api["url"], model, msgs, api["key_env"], api["budget"], temperature=temperature, max_tokens=api["max_tokens"],
                    reasoning_effort=api.get("reasoning_effort"), chat_template_kwargs=api.get("chat_template_kwargs"))
        text = o["text"]
        meta = dict(finish=o["finish"], 返ったモデル=o["model"], 考えた文=o["reasoning"], 使った数=o["usage"], 日時=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not text and o["finish"] == "length":
            return dict(判定="考えで切れた", 種類="", 本文="", 秒=round(time.time() - t0, 1), **meta)
        if str(o["finish"]).startswith("error"):
            return dict(判定="呼べない", 種類="", 本文=o.get("error", ""), 秒=round(time.time() - t0, 1), **meta)
    sec = time.time() - t0
    try:
        r = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        cut = meta.get("finish") == "length"                 # 本文はあるが長さで切れた物は、読めないと分けて数える
        return dict(判定="長さで切れた" if cut else "読めない", 種類="", 本文=text, 秒=round(sec, 1), **meta)
    if r.get("cannot_judge") in (True, "true", "True"):
        v = "判定できない"
    elif not isinstance(r, dict) or "has_error" not in r:      # JSON だが先生の形でない（指示文を受けずに返事を書いた など）
        return dict(判定="形が違う", 種類="", 本文=text, 秒=round(sec, 1), **meta)
    else:
        v = "指摘あり" if r.get("has_error") in (True, "true", "True") else "指摘なし"
    return dict(判定=v, 種類=str(r.get("kind") or "").strip() if v == "指摘あり" else "", 本文=text, 秒=round(sec, 1), **meta)


def auc(pos, neg):
    if not pos or not neg:
        return None
    a = np.array(list(neg) + list(pos), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=a.argsort().argsort() + 1.0) / cnt)[inv]
    return round(float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def split_kind(ans):
    """1 つの組（同じ行への何回かの答え）の割れ方: なし ／ あり・なしが割れた ／ 種類だけ割れた。読めないは除く。"""
    v = [a["判定"] for a in ans if a["判定"] not in NOT_ANSWER]
    if len(set(v)) > 1:
        return "あり・なしが割れた"
    k = [a["種類"] for a in ans if a["判定"] == "指摘あり"]
    if len(set(k)) > 1:
        return "種類だけ割れた"
    return "割れない"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--host", required=True)
    ap.add_argument("--runs", required=True, help="モデル@温度x回数 を , 区切り")
    ap.add_argument("--unsure", default="u1", choices=("off", "u1", "u2"))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--unstable", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--think", default="off", choices=("on", "off", "unset"), help="記録の回と同じにする（u1 の回は off）")
    ap.add_argument("--api", default="ollama", choices=("ollama", "openai"), help="openai = OpenAI 互換の口（--host は .../v1 まで）")
    ap.add_argument("--key_env", default="", help="--api openai の鍵を入れた環境変数の名前（鍵そのものは渡さない）")
    ap.add_argument("--max_calls", type=int, default=800, help="--api openai で 1 回の実行で口を呼ぶ回数の上限（再試行も数える）")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--reasoning_effort", default=None, help="考える型のモデルの考えの量（low など。口が受ける物だけ）")
    ap.add_argument("--no_thinking", action="store_true", help="chat_template_kwargs に enable_thinking: false を渡す")
    ap.add_argument("--lines", default=None, help="記録の md の代わりに、自作の行の JSON（[{流れ:[...], 行:...}]）で試す（つなぎの確認）")
    a = ap.parse_args()
    if a.lines:
        turns = [dict(会話=0, 往復=i + 1, 流れ=d["流れ"], 行=d["行"], 行の作り="自作", 説明="", 段=1, 記録の答え=None)
                 for i, d in enumerate(json.load(open(a.lines, encoding="utf-8")))]
    else:
        turns = read_turns(a.log)
    api = None
    if a.api == "openai":
        api = dict(url=a.host, key_env=a.key_env, budget=OC.Budget(a.max_calls), max_tokens=a.max_tokens, reasoning_effort=a.reasoning_effort,
                   chat_template_kwargs={"enable_thinking": False} if a.no_thinking else None)
    if a.limit:
        turns = turns[: a.limit]
    runs = []
    for s in a.runs.split(","):
        mdl, rest = s.rsplit("@", 1)
        tmp, n = rest.split("x")
        runs.append((mdl, float(tmp), int(n)))
    done = {}
    if os.path.exists(a.raw):                                # 続きから（同じ組の答えは読み直す）
        for ln in open(a.raw, encoding="utf-8"):
            d = json.loads(ln)
            done.setdefault((d["モデル"], d["温度"], d["会話"], d["往復"]), []).append(d)
    res = dict(行=len(turns), 段の無い行=sum(t["段"] is None for t in turns), 組=[f"{m}@{tp}x{n}" for m, tp, n in runs], 秒={})

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    t0 = time.time()
    with open(a.raw, "a", encoding="utf-8") as fo:
        for mdl, tp, n in runs:
            t1 = time.time()
            for k, t in enumerate(turns):
                key = (mdl, tp, t["会話"], t["往復"])
                have = done.setdefault(key, [])
                while len(have) < n:
                    try:
                        j = judge(a.host, mdl, t, a.unsure, tp, {"on": True, "off": False, "unset": None}[a.think], api)
                    except OC.BudgetOver as e:
                        print("止めた:", e, flush=True)
                        res["止めた"] = str(e)
                        save()
                        return
                    d = dict(モデル=mdl, 温度=tp, 会話=t["会話"], 往復=t["往復"], 回=len(have) + 1, **j)
                    fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                    fo.flush()
                    have.append(d)
                if (k + 1) % 40 == 0:
                    print(mdl, tp, k + 1, f"{time.time() - t1:.0f} 秒", flush=True)
            res["秒"][f"{mdl}@{tp}"] = round(time.time() - t1)
            save()
    # 数える
    by = {(m, tp): {(t["会話"], t["往復"]): done[(m, tp, t["会話"], t["往復"])][:n] for t in turns} for m, tp, n in runs}
    within = {}
    unstable = {}
    within_split = {}
    for (m, tp), D in by.items():
        sk = {key: split_kind(ans) for key, ans in D.items()}
        why_of = {(t["会話"], t["往復"]): t["行の作り"] for t in turns}
        c = Counter(sk.values())
        calls = [x for ans in D.values() for x in ans]
        within[f"{m}@{tp}"] = dict(
            割れ方={k: int(v) for k, v in c.items()}, 割れた行の割合=round(1 - c["割れない"] / max(1, len(sk)), 4),
            答えの内訳=dict(Counter(x["判定"] for x in calls)), 種類の内訳=dict(Counter(x["種類"] for x in calls if x["判定"] == "指摘あり")),
            割れた行の行の作り=dict(Counter(why_of[k] for k, v in sk.items() if v != "割れない").most_common(10)),
            一回の秒の中央値=float(np.median([x["秒"] for x in calls])) if calls else None)
        within_split[(m, tp)] = {k for k, v in sk.items() if v != "割れない"}
        for key, v in sk.items():
            if v != "割れない":
                unstable.setdefault(key, []).append(f"{m}@{tp}: {v}")
    res["1_同じモデルの中"] = within

    def major(ans):
        v = Counter(x["判定"] for x in ans if x["判定"] not in NOT_ANSWER).most_common(1)
        if not v:
            return "読めない", ""
        k = Counter(x["種類"] for x in ans if x["判定"] == "指摘あり").most_common(1)
        return v[0][0], (k[0][0] if v[0][0] == "指摘あり" and k else "")
    models = sorted({m for m, _, _ in runs})
    low = {m: by[(m, min(tp for mm, tp, _ in runs if mm == m))] for m in models}
    between = {}
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            same_v = same_k = both = 0
            split_keys = set()
            for key in low[m1]:
                v1, k1 = major(low[m1][key])
                v2, k2 = major(low[m2][key])
                same_v += v1 == v2
                if v1 == v2 == "指摘あり":
                    both += 1
                    same_k += k1 == k2
                if v1 != v2 or (v1 == v2 == "指摘あり" and k1 != k2):
                    split_keys.add(key)
                    unstable.setdefault(key, []).append(f"{m1} と {m2}: {'あり・なし' if v1 != v2 else '種類'}")
            within_keys = set().union(*[ks for (m, _), ks in within_split.items() if m in (m1, m2)])
            between[f"{m1} と {m2}"] = dict(判定が合う割合=round(same_v / max(1, len(low[m1])), 4), 両方ありの行=both,
                                          種類が合う割合=round(same_k / max(1, both), 4) if both else None, 割れた行=len(split_keys),
                                          同じモデルの中で割れた行との重なり=len(split_keys & within_keys),
                                          組の表=dict(Counter(f"{major(low[m1][k])[0]}／{major(low[m2][k])[0]}" for k in low[m1])))
    res["2_モデルの間"] = between
    # 3. 揺れた行と、ボット自身の違和感
    tmap = {(t["会話"], t["往復"]): t for t in turns}
    with open(a.unstable, "w", encoding="utf-8") as fo:
        for key, why in sorted(unstable.items()):
            t = tmap[key]
            fo.write(json.dumps(dict(会話=key[0], 往復=key[1], 段=t["段"], 割れ方=why, 記録の答え=t["記録の答え"], 行=t["行"], 行の作り=t["行の作り"],
                                     流れ=t["流れ"], ボット自身=own_odd(t["説明"]),
                                     答え={f"{m}@{tp}": [(x["判定"], x["種類"]) for x in by[(m, tp)][key]] for m, tp, _ in runs}),
                                ensure_ascii=False) + "\n")
    cmp_ = {}
    for nm in ("点", "行づくり", "作る途中の違和感", "P指摘の最大"):
        u = [own_odd(tmap[k]["説明"])[nm] for k in tmap if k in unstable]
        s = [own_odd(tmap[k]["説明"])[nm] for k in tmap if k not in unstable]
        u, s = [x for x in u if x is not None], [x for x in s if x is not None]
        cmp_[nm] = dict(揺れた行の平均=round(float(np.mean(u)), 4) if u else None, 揺れない行の平均=round(float(np.mean(s)), 4) if s else None,
                        揺れた行が大きいAUC=auc(u, s))
    res["3_揺れた行"] = dict(数=len(unstable), 割合=round(len(unstable) / max(1, len(turns)), 4),
                          行の作り=dict(Counter(tmap[k]["行の作り"] for k in unstable).most_common(10)), ボット自身の値=cmp_,
                          記録の答えの内訳_揺れた行=dict(Counter(tmap[k]["記録の答え"] for k in unstable)))
    res["秒"]["全体"] = round(time.time() - t0)
    save()
    print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False)[:4000], flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
