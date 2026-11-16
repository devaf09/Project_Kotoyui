"""I48u: 先生（会話の相手役とは別の、見立て役の LLM）の指示の型を、ダメ出しから促しの型に書き直す（ロードマップ 11）。
今の型（chat_partner.TEACHER）は残し、chat_partner の --teacher_prompt で切り替える。

ねらい: 「指摘されたか」が行を見分ける信号になること（今の型は返事の 84% に何か言う。I48q）。
- 先生はまず「この返事はこのままでよいか」を決める（判定 = 良 / そのまま / 直す）。
- 直す時だけ、どこをどう言えばよかったかを 1 つ、促しの形で言う。
- 良い返事（相手の話を受けた、話を結んだ、中身があった）は 良 にしてよい。
- 出力は決まった形（判定・種類・一言）。I48p の数え違い（先生の地の文の「〜と言っているから」）が起きないよう、分け方は種類の欄だけで読む。

型は 3 通り（どれが印と重なるかを、記録の掛け直しで比べて選ぶ。割合の目標は決め打ちにしない）:
  p2a: 判定の線を「相手が戸惑う・会話が止まるほどの所があるか」に置く。
  p2b: p2a ＋ 良の決め方を書く（受けた・結んだ・中身がある）。
  p2c: p2b ＋ 「迷ったら そのまま」。
"""
import json

HEAD = (
    "あなたは、日本語の会話を練習しているボットを見守る先生です。ボットは少しずつ話せるようになっている途中です。\n"
    "「会話の流れ」と「ボットの返事」を読み、まず、この返事をこのままにしてよいかを決めます。\n"
)
LINE = (
    "判定の決め方:\n"
    "- 直す: 相手が戸惑う、または会話が止まってしまうほどの所が 1 つでもある時だけ。文法の誤り、ふつうは言わない言葉づかい、"
    "前の相手の行に答えていない、など。\n"
    "- そのまま: 上の所が無い時。少し短い・少しぎこちない、くらいなら そのまま にする。\n"
)
GOOD = (
    "- 良: そのまま のうち、相手の話をきちんと受けている、前に出た話と今の話を結んでいる、自分の中身（気持ち・経験・考え）がある、"
    "のどれかが言える時。\n"
)
DOUBT = "- 迷ったら そのまま にする。\n"
RULES = (
    "一言の決まり:\n"
    "- 直す 時だけ、どこをどう言えばもっと伝わるかを 1 つだけ、促しの形で言う（「〜と言うと、もっと伝わるよ」）。ダメ出しの言い方はしない。\n"
    "- 良 の時は、良かった所を短く褒める。そのまま の時は一言を空にする。\n"
    "- 直す 時は、おかしい所をボットの返事からそのまま抜き出し（odd_part）、自然な返事の例（suggested_reply）を 1 つ出す。\n"
    "出力は次の JSON だけ:\n"
    '{"verdict": "良 か そのまま か 直す", "kind": "文法 か 言い方 か つながり か 足りない（直す の時だけ。ほかは空）", '
    '"odd_part": "直す 時の、ボットの返事の中のおかしい所（ほかは空）", "suggested_reply": "直す 時の、自然な返事の例（ほかは空）", '
    '"message": "ボットへの一言（そのまま の時は空）"}\n'
    "kind の決め方: 文法 = 助詞・活用・語の組み合わせの誤り。言い方 = 意味は通るが、ふつうはそう言わない言葉づかい。"
    "つながり = 文はふつうだが、前の相手の行に答えていない・話が飛ぶ。足りない = 文はふつうだが、会話が止まるほど短い・素っ気ない。\n"
)
PROMPTS = {
    "p2a": HEAD + LINE + RULES,
    "p2b": HEAD + LINE + GOOD + RULES,
    "p2c": HEAD + LINE + GOOD + DOUBT + RULES,
}
VERDICTS = ("良", "そのまま", "直す")


def parse(text, reply):
    """新しい型の出力を、今の型と同じ欄（has_error・odd_part・kind・suggested_reply・intervention_message）と verdict に直す。
    読めない時・見立てが返事と食い違う時は (None, 理由)。そのまま は ev を返し、呼ぶ側で経験に入れない。"""
    j = text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""
    try:
        r = json.loads(j)
    except Exception:
        return None, "読めない"
    v = str(r.get("verdict") or "").strip()
    v = next((x for x in VERDICTS if x in v), None)
    if v is None:
        return None, "判定が無い"
    odd, sug = str(r.get("odd_part") or "").strip("「」。 "), str(r.get("suggested_reply") or "").strip()
    ev = dict(verdict=v, has_error=v == "直す", kind=str(r.get("kind") or "").strip() if v == "直す" else "",
              odd_part=odd if v == "直す" else "", suggested_reply=sug if v == "直す" else "",
              intervention_message=str(r.get("message") or "").strip())
    if v == "直す":
        if not odd or odd not in reply:
            return None, "返事に無い所"
        if sug.strip("。") == reply.strip("。"):
            return None, "お手本が同じ"
    return ev, ""
