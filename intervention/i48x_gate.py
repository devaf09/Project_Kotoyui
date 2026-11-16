"""I48x: 門（ロードマップ 53）。倫理は処罰でなく門として持つ。

1. 深刻な相談の門（--gate）: 命や健康に関わる話を見つけたら、型の板も行づくりも通さず、固定の返事を返す。
   - 固定の返事の文は持ち主と決める。今は仮の文を設定ファイル（i48x_gate_reply.txt の 1 行）に置く。
   - ここは学びで薄れない・暗くならない・点で負けない（bot の側で、この行は経験に入れない）。
   - 見つけ方: 安全のための例外として、手で置いた語の目印（CRISIS）を使う。見逃しを減らす側に倒す（拾いすぎてもよい）。
     near（意味の波で近い語）を渡すと、目印の語に近い語も拾う。
2. 白銀律の門（--silver）: 相手の行で bot が指摘された・拒まれた時、その相手の行の所（名詞＋助詞＋述語）を「自分がされて嫌だった形」として覚え、
   bot が同じ所を言う前に暗くする。相手の善意の行（おせっかい）は数えない（指摘・拒みの行だけ）。chat_bot3 の dark_line で使う。
3. 「今は言ってよい時か」の門（感情の読み手ができてから）: mood_ok は今はいつも真（口だけ）。
"""
import os

# 目印（原形・表記の一部）。安全のための例外として手で置く。見逃しを減らす側に倒す
CRISIS = ("死にたい", "死にたく", "死のう", "死ぬしか", "消えたい", "消えてしまいたい", "いなくなりたい", "生きていたくない", "生きてる意味",
          "生きている意味", "生きる意味が", "自殺", "自傷", "リストカット", "リスカ", "首を吊", "飛び降り", "オーバードーズ", "殺される",
          "殺したい", "殴られ", "暴力を", "虐待", "追い詰め", "もう限界", "助けて", "息ができない", "胸が苦しい", "倒れそう", "救急車",
          "眠れない日が", "何日も食べ")
REPLY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "i48x_gate_reply.txt")


def fixed_reply(path=REPLY_FILE):
    with open(path, encoding="utf-8") as f:
        return next(x.strip() for x in f if x.strip() and not x.startswith("#"))


def crisis(text, toks=None, near=None):
    """深刻な相談の目印があるか。text の中の目印の文字列、または（near があれば）目印の語に意味の波で近い語。"""
    t = text.replace(" ", "").replace("　", "")
    if any(m in t for m in CRISIS):
        return True
    if near is not None and toks:
        keys = ("死ぬ", "自殺", "消える", "殺す", "暴力", "虐待", "限界")
        return any(near.near(tk["lemma"], k) and tk["lemma"] not in ("消す",) for tk in toks for k in keys)
    return False


def mood_ok(state=None):
    """「今は言ってよい時か」（相手の怒り・恐れが高い時は、促さず受けるだけ）。感情の読み手ができるまでは、いつも真。"""
    return True
