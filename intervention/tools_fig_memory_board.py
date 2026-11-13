"""記憶板の仕組みの図を描く（行の板 → 分・時・日・週、大事な板、読む道）。

使い方: python intervention/tools_fig_memory_board.py <出力.png>
  日本語のフォントは IPA P ゴシック（下の addfont の場所）を使う。別の場所なら書き換える。
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fm.fontManager.addfont("/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf")
plt.rcParams["font.family"] = "IPAPGothic"
BG, INK, SUB, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
fig, ax = plt.subplots(figsize=(14, 8.6), dpi=200)
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 140); ax.set_ylim(0, 86); ax.axis("off")

def box(x, y, w, h, title, body, col, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=col, lw=1.6))
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=fs, color=col, weight="bold", va="top")
    ax.text(x + 1.2, y + h - 5.0, body, fontsize=8.6, color=INK, va="top", linespacing=1.45)

def arrow(a, b, txt="", col=SUB, off=(0, 0), rad=0.0, fs=8.2, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], txt, fontsize=fs, color=col, ha=ha, va="center",
                linespacing=1.35)

ax.text(1, 84, "記憶板の仕組み（案、2026-11-11）", fontsize=14, color=INK, weight="bold", va="top")
ax.text(1, 80.2, "会話の行を板にし、時間の段ごとに和で畳む。畳む時は、期待より明るい所だけが残る（忘れる）。大事なことは別の板に薄めずに残す。",
        fontsize=9.5, color=SUB, va="top")

# 会話
box(1, 58, 26, 17, "会話の行", "人「明日は歯医者なんだ」\nボット「明日は歯医者なの？」\n人「うん、10 時から」\n…", GRAY)
# 行の板
box(33, 58, 30, 17, "行の板（秒 = 1 行）", "所: (明日, は, 歯医者)\n話題の名詞: 明日・歯医者\n誰: 人　時刻: 10:02\n結果: 良 / 指摘", BLUE)
arrow((27.6, 66.5), (32.4, 66.5), "読む", off=(0, 1.6))

# 段
levels = [("分（1 つの話題）", "話題の名詞が替わるまでの行の板の和"),
          ("時（1 つの会話）", "試し: 1 会話 = 1 時（将来は実時間）"),
          ("日", "試し: 20 会話 = 1 日"),
          ("週", "7 日の和")]
ys = [44, 31, 18, 5]
for (t, b), y in zip(levels, ys):
    box(33, y, 30, 10, "記憶板 " + t, b, BLUE, fs=10)
arrow((48, 57.6), (48, 54.6), "")
for y0, y1 in zip(ys[:-1], ys[1:]):
    arrow((48, y0 - 0.4), (48, y1 + 10.4), "")
ax.text(64.5, 34, "下の段が閉じたら\n（話題が替わる・会話が終わる・日が替わる）\n下の段の板を足して、上の段を 1 枚作る\n（親の語板 = 子の語板の和、と同じ）",
        fontsize=8.4, color=BLUE, va="center", linespacing=1.45)
ax.text(64.5, 21, "忘れる: 畳む時、|観測 − 期待| の大きい鍵だけ残す\n期待 = ふだんの会話で、その段の長さに出る見込み\n相槌・いつもの話は消え、よく出た話題が明るく残る",
        fontsize=8.4, color=BLUE, va="center", linespacing=1.45)

# 大事な板
box(103, 44, 36, 31, "大事な板（相手ごと、薄めない）",
    "名前: 「私は〜」「〜って呼んで」・人名\n予定: 時の語（明日・来週・〜時）+ 述語\n　→ 過ぎたら「済んだ」\n頼まれごと: 「〜して」「お願い」「覚えておいて」\n相手のこと: 相手が自分について言った所\n\n例: 相手 A ／ 予定: 明日 10 時 歯医者\n\n置き場: PC の非公開フォルダ（JSON）\n暗号化はクラウドに置く時だけ", ORANGE)
arrow((63.6, 70), (102.4, 70), "目印で抜き出す\n（将来は、会話の中で大事かを自分で判断）", col=ORANGE, off=(0, 3.2))

# 読む
box(103, 5, 36, 33, "読む（問いに答える時）",
    "「昨日何話した？」「前にも言ったっけ」\n「明日の予定、覚えてる？」\n\n1. 大事な板を、相手と問いの語で探す\n2. 記憶板を、新しい段から順に\n　（分 → 時 → 日 → 週）、問いの語と\n　近い鍵を探す（自分の板の近さと同じ）\n3. 見つかった段で言い方を替える\n　さっき・今日・昨日・この前", GREEN)
arrow((63.6, 8), (102.4, 10), "古い段ほど、よく出た所だけ", col=GREEN, off=(0, -2.6))
arrow((121, 43.6), (121, 38.6), "", col=GREEN)

# 自分の板
box(1, 18, 26, 26, "自分の板（あとで）", "その日の記憶板の中で、自分について\n言った所（「今日はおしゃべりしてた」）\nを、その日の自分の板に入れる。\n同じ日に反対のことを言いそうなら\n違和感の器で気づく。", GRAY)
arrow((32.4, 23), (27.6, 28), "", col=GRAY)

fig.savefig(sys.argv[1], facecolor=BG, bbox_inches="tight")
