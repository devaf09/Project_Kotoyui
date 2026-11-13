"""I35y〜I38 の会話の記録（i35y/*.md、非公開）から、ボット自身についての問いに、ボットが答えられなかったものを拾って一覧にする。
書き出し: docs/答えられない問い_wn.md（非公開。commit しない）と i35y/答えられない問い_wn.jsonl
対象: Gemma が話し相手の回（e, e2, f, f2, g, h, i, j, k, l）。
ボット自身についての問い: 話し相手の行で「？」で終わり（「〜って？」の聞き返しは除く）、主語がボット（君・そっち・あなた、
  または主語なしで「何してるの」「どこか行きたい」「お腹空いてない」「何が好き」のようにボットに向いた形）。
  猫・犬・それ など、ボット以外が主語の問いは除く。
答えられなかった、とみるもの（すぐ後のボットの行で決める）:
  知らない   「いや、分からん。」（自分の板が空なので、知らないと返す）
  板に無い   自分のこと（自分の板に (…) が無い）で否定を返した
  聞き返し   語だけを「？」で聞き返した（「猫？」）
  属性を聞く 話題の語の属性を問いで返した（「はい、〜の？」）
  相槌だけ   「はい。」「うん。」だけ
  先生       先生が「質問に答え」「答えてあげ」などと言った（上のどれでもない行づくりの行など）
  行づくり（先生なし） 先生のいない回（e, e2）で、行づくりの行を返した（答えになっているかは確かめていない。数には入れず別に数える）
使い方: python intervention/analysis/i35y_unanswered.py  （リポジトリの直下から。記録は非公開の置き場 KOTOYUI_PRIVATE から読む）
"""
import collections
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P = os.environ.get("KOTOYUI_PRIVATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Project_Kotoyui_private"))   # 非公開の置き場（記録）
Y = os.path.join(P, "i35y")
RUNS = [
    ("e", "i35y_gemma_wn.md"), ("f", "i35z_gemma_teacher_gemma_wn.md"),
    ("e2", "i35y_gemma_i37a_wn.md"), ("f2", "i35z_gemma_teacher_gemma_i37a_wn.md"),
    ("g", "i35z_gemma_teacher_gemma_good_i37b_wn.md"), ("h", "i35z_gemma_teacher_gemma_good_i37c_wn.md"),
    ("i", "i35z_gemma_teacher_gemma_good_flow_i37d_wn.md"), ("j", "i35z_gemma_teacher_gemma_good_flow_novelty_i38a_wn.md"),
    ("k", "i35z_gemma_teacher_gemma_good_flow_novelty_i37e_wn.md"), ("l", "i35z_gemma_teacher_gemma_good_flow_novelty_base_i37f_wn.md"),
]
NO_TEACHER = {"e", "e2"}
ASKBACK = re.compile(r"(って|のって)[？?]\s*$")
YOU = re.compile(r"君|きみ|そっち|あなた|お前")
SELF_FORMS = re.compile(
    r"何してる|何して(た|る|い)|何した|何する|何しよう|何して過ごす|どこか(に|へ)?(行き|お出かけ|出かけ)|行きたい|したい(こと|もの)?|"
    r"何が好き|どんな.*好き|好きな|好き[？?]|嫌い|お腹|空いて|食べた|食べたい|飲む|飲みたい|飲んで|予定|疲れ|元気|眠|寒くない|暑くない|"
    r"名前|住んで|飼って|行った|出かけた|見た[？?]|読んだ|遊んだ|決まって|ある[？?]|いる[？?]|どう[？?]|どうだった|思う[？?]")
OTHER_SUBJ = re.compile(r"(猫|犬|ペット|それ|あれ|これ|この|その|あの|お店|天気|空|雨|花|鳥|町|景色|友達|お母さん|お父さん|妹|弟|兄|姉)(ちゃん)?(は|が|って|、|の)")
TOPICS = [  # 上から順に当てる
    ("名前", r"名前|なんて呼|呼んでる"),
    ("体の状態", r"お腹|空いて|空かない|疲れ|眠|元気|寒くない|暑くない|痛|大丈夫"),
    ("家・ペット", r"家|住ん|部屋|おうち|ペット|飼|猫|犬|家族"),
    ("行った所", r"行った|出かけた|行ってきた|どこ(に|へ)行っ|旅行"),
    ("予定・したいこと", r"予定|したい|行きたい|やりたい|食べたい|飲みたい|する(の)?[？?]|するの|しよう|つもり|明日|週末|今度|これから|次は|"
                         r"(行く|出かける|お出かけする|過ごす|食べる|飲む)の?[？?]"),
    ("今していること", r"何してる|何して(た|る|い)|何した|何してた|今日は何|今は|さっき|最近|食べた|飲んだ|買った|見た|読んだ"),
    ("好み", r"好き|嫌い|お気に入り|どんな"),
]


def last_question(q):
    """前置きの文を除いた、問いの文（最後の「？」を含む文）。"""
    parts = [p for p in re.split(r"(?<=[。！!])", q) if p.strip()]
    qs = [p for p in parts if re.search(r"[？?]", p)]
    return (qs[-1] if qs else q).strip()


def topic(q):
    s = last_question(q)
    for k, pat in TOPICS:
        if re.search(pat, s):
            return k
    return "その他"


def about_bot(q):
    if YOU.search(q):
        return True
    if OTHER_SUBJ.search(q):
        return False
    return bool(SELF_FORMS.search(q))


def turns(md):
    t = open(os.path.join(Y, md), encoding="utf-8").read()
    for m in re.finditer(r"## 会話 (\d+)\n(.*?)(?=\n## 会話 |\Z)", t, re.S):
        ci, body = int(m.group(1)), m.group(2)
        cur = None
        for line in body.splitlines():
            if line.startswith("- 相手: "):
                if cur:
                    yield cur
                cur = dict(conv=ci, partner=line[len("- 相手: "):].strip(), bot="", why="", teacher="")
            elif line.startswith("- ボット: ") and cur is not None:
                mm = re.match(r"- ボット: \*\*(.*?)\*\* （(.*)）\s*$", line)
                cur["bot"], cur["why"] = (mm.group(1), mm.group(2)) if mm else (line[7:], "")
            elif line.startswith("  - 先生") and cur is not None:
                cur["teacher"] += line.strip()[2:] + " "
        if cur:
            yield cur


def short_why(w):
    w = re.sub(r"（点 [^）]*）|（新規性[^）]*）|、[0-9.]+ 秒$", "", w)
    w = re.sub(r" ／ 「[^」]*」→ 違和感「[^」]*」（[^）]*）", "", w)       # 同じ違和感の並びを 1 つに
    w = re.sub(r"（試し: [^）]*）", "（試し）", w)
    return w[:160] + ("…" if len(w) > 160 else "")


def short_teacher(te):
    te = re.sub(r"\s*／ 経験:.*$", "", te.strip())
    te = re.sub(r"^先生（段 \d）: ", "", te)
    return te[:120] + ("…" if len(te) > 120 else "")


def kind(t, run):
    b, w, te = t["bot"].strip(), t["why"], t["teacher"]
    if "知らないと返す" in w or "分からん" in b:
        return "知らない"
    if "自分のこと（自分の板に" in w:
        return "板に無い"
    if re.fullmatch(r"[^、。]{1,8}？", b):
        return "聞き返し"
    if "の属性で" in w and b.endswith("？"):
        return "属性を聞く"
    if b in ("はい。", "うん。", "ああ。"):
        return "相槌だけ"
    if re.search(r"質問に答え|答えてあげ|答えるのが自然|答えるほうが|答えるといい|答えてみよう|答えるように", te):
        return "先生"
    if run in NO_TEACHER and "行づくり" in w:
        return "行づくり（先生なし）"
    return ""


rows, asked = [], 0
for run, md in RUNS:
    if not os.path.exists(os.path.join(Y, md)):
        continue
    for t in turns(md):
        q = t["partner"]
        if not re.search(r"[？?]", q) or ASKBACK.search(q) or not about_bot(q):
            continue
        asked += 1
        k = kind(t, run)
        if k:
            rows.append(dict(run=run, conv=t["conv"], q=q, bot=t["bot"], kind=k, topic=topic(q),
                             why=short_why(t["why"]), teacher=short_teacher(t["teacher"])))

un = [r for r in rows if r["kind"] != "行づくり（先生なし）"]
made = [r for r in rows if r["kind"] == "行づくり（先生なし）"]
by_kind = collections.Counter(r["kind"] for r in rows)
TOP = [k for k, _ in TOPICS] + ["その他"]
by_topic = collections.Counter(r["topic"] for r in un)


def norm(q):
    return re.sub(r"^(うん|ううん|そっか|そうだね|いいね|いいなー?|わかる|本当だね|あはは|えー|へえ|なるほど)[、。！!ー]*", "", re.sub(r"[\s]", "", q))


out = ["# 答えられない問い（Gemma の回、非公開）", "",
       "I35y〜I38（2026-11-10）の Gemma が話し相手の回（e, e2, f, f2, g, h, i, j, k, l）から、ボット自身についての問いに、ボットが答えられなかったもの。",
       "プロフィール（自分の板）を作る材料。拾い方は `experiments/i35y_unanswered.py` の説明を見る。commit しない。", "",
       f"- ボット自身についての問い: {asked} 回",
       f"- 答えられなかった: {len(un)} 回（{len(un) / max(1, asked):.0%}）。ほかに、先生のいない回（e, e2）で行づくりの行を返した {len(made)} 回（答えになっているかは確かめていない）", "",
       "## 答えられなかった形", "", "| 形 | 回数 |", "|---|---|"]
out += [f"| {k} | {v} |" for k, v in by_kind.most_common()]
out += ["", "## 型ごとの数", "", "| 型 | 答えられなかった | 異なる問い |", "|---|---|---|"]
uniq_by_topic = {tp: collections.OrderedDict() for tp in TOP}
for r in un:
    uniq_by_topic[r["topic"]].setdefault(norm(r["q"]), []).append(r)
out += [f"| {tp} | {by_topic[tp]} | {len(uniq_by_topic[tp])} |" for tp in TOP]
out += ["", "## 一覧（型ごと。同じ問いはまとめ、多く出た順）", ""]
for tp in TOP:
    if not uniq_by_topic[tp]:
        continue
    out += [f"### {tp}（{by_topic[tp]} 回、異なる問い {len(uniq_by_topic[tp])}）", "",
            "| 問い | 回数 | ボットの返事 | 形 | ボットの理由（縮め） | 先生（縮め） | 回・会話 |", "|---|---|---|---|---|---|---|"]
    for key, rs in sorted(uniq_by_topic[tp].items(), key=lambda kv: -len(kv[1])):
        r = rs[0]
        where = "、".join(f"{x['run']}-{x['conv']}" for x in rs[:6]) + ("…" if len(rs) > 6 else "")
        bots = " ／ ".join(dict.fromkeys(x["bot"] for x in rs[:3]))
        cell = lambda s: s.replace("|", "／")
        out.append(f"| {cell(r['q'])} | {len(rs)} | {cell(bots)} | {r['kind']} | {cell(r['why'])} | {cell(r['teacher']) or '—'} | {where} |")
    out.append("")
os.makedirs(os.path.join(P, "docs"), exist_ok=True)
open(os.path.join(P, "docs", "答えられない問い_wn.md"), "w", encoding="utf-8").write("\n".join(out))
with open(os.path.join(Y, "答えられない問い_wn.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("about-bot questions", asked, "unanswered", len(un), "made(no teacher)", len(made))
print("kinds", dict(by_kind.most_common()))
for tp in TOP:
    print(f"== {tp} {by_topic[tp]} 回 / 異なる {len(uniq_by_topic[tp])}")
    for key, rs in sorted(uniq_by_topic[tp].items(), key=lambda kv: -len(kv[1]))[:15]:
        r = rs[0]
        print(f"  {len(rs)}| {r['q']} → {r['bot']} 〔{r['kind']}〕 {r['run']}-{r['conv']}")
