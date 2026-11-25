"""I49: 思考のループの図（矢印の状態と問題点、2026-11-23 夜の作り）を描く。数は入れていない、手で書いた図。
python tools/past/fig_i49_loop.py docs/images/i49_loop_arrows.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
FONT="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
fm.fontManager.addfont(FONT); plt.rcParams["font.family"]=fm.FontProperties(fname=FONT).get_name()
C=dict(ok="#2e7d4f",half="#d98200",read="#00798c",assoc="#8e6bbf",build="#c9861a",out="#d1495b",wait="#8a8a8a",ink="#30323d",soft="#f3f1ec",line="#b9b6ae",warn="#c0392b",odd="#2e7d4f")
fig=plt.figure(figsize=(12,19.2)); a=fig.add_axes([0,0,1,1]); a.set_xlim(0,120); a.set_ylim(-60,132); a.axis("off")
a.text(2,129,"思考のループの図（11-23 夜の作り）。矢印の状態と問題点",fontsize=15,color=C["ink"],va="center")
a.text(2,124.9,"1 歩に 1 つだけ動く。思考のループが 1 歩進むたびに、感情・状態のループ（器）が 1 刻み進む。\n赤い番号は下の問題点、英字は下の矢印の一覧。前の図から、矢印の状態と、数えの部品・介入・当たり外れを足した。",fontsize=9.5,color="#555",va="center")
def box(x,y,w,h,t="",fc=C["soft"],ec=C["line"],fs=9.3,tc=C["ink"],lw=1.2,ls="-"):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec=ec,lw=lw,linestyle=ls))
    if t: a.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc,linespacing=1.5)
def arr(p,q,col=C["ink"],rad=0,ls="-",lw=1.5):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=13,lw=lw,color=col,connectionstyle=f"arc3,rad={rad}",linestyle=ls))
def num(x,y,n):
    a.add_patch(Circle((x,y),1.55,fc=C["warn"],ec="white",lw=1,zorder=5)); a.text(x,y-.05,str(n),ha="center",va="center",color="white",fontsize=9.5,zorder=6)
def tag(x,y,ch,col):
    a.add_patch(FancyBboxPatch((x-1.3,y-1.3),2.6,2.6,boxstyle="round,pad=0.1,rounding_size=.5",fc="white",ec=col,lw=1.6,zorder=7)); a.text(x,y-.05,ch,ha="center",va="center",color=col,fontsize=9.5,zorder=8,fontweight="bold")
def pill(x,y,t,col,w=9):
    a.add_patch(FancyBboxPatch((x,y-1.6),w,3.2,boxstyle="round,pad=0.15,rounding_size=.9",fc=col,ec="none")); a.text(x+w/2,y,t,ha="center",va="center",color="white",fontsize=9.5)
# 読み手
box(2,96,21,21,"場（3 人で読む席）\n\n本のページ\n先生 2 人の番\n聞き返し役の先生の返事\n\n語を 1 つずつ渡す\n（語のベクトルと印）\n自分の番 = 文の終わりの印",fs=8.6)
# 思考のループ
box(27,62,66,58,fc="#ffffff",ec=C["ink"],lw=1.6)
a.text(29,117.3,"思考のループ",fontsize=11.5,color=C["ink"],va="center")
# 入れ物
box(30,96,26,17,fc=C["soft"])
a.text(43,110.6,"入れ物（ばらばらの材料置き場）",ha="center",fontsize=9.3)
pill(31.5,105.6,"語",C["read"],6); a.text(38.5,105.6,"聞いた語",va="center",fontsize=8.6)
pill(31.5,101.6,"連想",C["assoc"],6); a.text(38.5,101.6,"連想で見つかった語",va="center",fontsize=8.6)
pill(31.5,97.8,"印",C["build"],6); a.text(38.5,97.8,"文の終わり・過去 など",va="center",fontsize=8.6)
# 連想
box(62,98,28,15,"連想（指揮者の手前）\n入れ物（空なら内面）に近い語を 5 つ引き、\n「材料が不足」をいちばん減らす\n1 つだけを入れ物に足す",fc="#f1ebf8",ec=C["assoc"],fs=8.6)
arr((61.6,105),(56.6,105),C["assoc"])
# 指揮者
box(30,65,60,27,fc="#fbfaf7",ec=C["ink"])
a.text(32,89.4,"指揮者（違和感を見て、次の 1 つを選ぶ）",fontsize=9.8,va="center")
a.text(32,85.9,"違和感「材料が不足」= 入れ物の中身の向きと、内面の向きのずれ。\n閾値は置かず、連想で減らなくなったら消えたとみなす。",fontsize=8.2,va="center",color="#555",linespacing=1.4)
steps=[("連想",C["assoc"],"不足が減るうちは、連想で足す"),("組立",C["build"],"減らなくなったら、組立の口へ"),("出力",C["out"],"組立に違和感なし → 声に出す。入れ物は空に"),("読む",C["read"],"違和感あり → 次の 1 語を読む"),("待つ",C["wait"],"語も無い → 休む。語が届くか、連想が効けば戻る")]
for i,(n,col,t) in enumerate(steps):
    y=80.6-i*3.5; pill(32,y,n,col,8); a.text(41.5,y,t,va="center",fontsize=8.8)
a.text(88.5,74,"上から順に\n当てはまる物を\n1 つ",ha="right",va="center",fontsize=8,color="#777")
# 組立の口
box(97,93,21,23,"組立の口（差し込み）\n\n選ぶ口: 材料から 3 つ選ぶ\n（いつもは拾われやすさ、\n言い直しは言いたかった\n向きへの近さ）\n行づくり: 聞いた順に並べ、\n助詞・語尾を入れる\n返す違和感: 途中・\n材料が少ない・もう言った",fc="#fdf3e1",ec=C["build"],fs=8.2)
arr((90.6,88),(96.6,96),C["build"],rad=-.15); arr((96.6,99),(90.6,90.5),C["build"],rad=.25)
a.text(91.5,95,"行と違和感が戻る",fontsize=8,color=C["build"],ha="right",va="center")
# 声
box(97,61.5,10,7,"外（声）\n行の記録",fs=8.4)
box(111.5,61.5,6.5,7,"会話",fs=8.8,ec=C["warn"],ls=":")
arr((107.5,65),(111.1,65),C["warn"],ls=":",lw=1.6); tag(109.3,67.4,"J",C["warn"])
arr((90.6,68),(96.6,65.5),C["out"]); a.text(95.05,69.6,"出力",fontsize=7.6,color=C["out"],ha="center")
box(97,73,21,16,"聞き返し役の先生\n（手元の言葉のモデル）\n\n分かった → 次を言う\n分からない → 聞き返しと\n「こういう意味？」",fc="#e8f1f3",ec=C["read"],fs=8.4)
arr((102,68.9),(102,72.6),C["out"]); arr((96.6,80),(90.6,80),C["read"],ls="--",lw=1.2)
a.text(95.05,83.6,"返事を\n聞く",fontsize=7.2,color=C["read"],ha="center")
# 読み手 → 入れ物
arr((23.6,108),(29.6,106),C["read"]); a.text(25.1,110.6,"読む",fontsize=7.6,color=C["read"],ha="center")
arr((29.6,72),(14,95.4),C["read"],rad=-.2,ls="--",lw=1.1); a.text(10.5,80,"「次の語」と\n呼ぶのはループ",fontsize=8.2,color=C["read"],ha="center")
# 器
box(27,30,50,24,fc="#ffffff",ec=C["ink"],lw=1.6)
a.text(29,51.4,"感情・状態のループ（器）",fontsize=11.5,va="center")
for i,col in enumerate(["#d1495b","#edae49","#00798c","#30323d"]):
    a.plot([33,60],[46-i*3.4]*2,color=col,lw=3,solid_capstyle="round")
a.text(31.6,46,"速",ha="right",va="center",fontsize=8.5); a.text(31.6,35.8,"遅",ha="right",va="center",fontsize=8.5)
a.text(62,43.5,"波 20 本\n動きの量・感度・\nいつもの外れ",fontsize=8.6,va="center")
a.text(29,32.2,"読んだ語と、自分の声を聞く。何も無い歩は、時間だけ進む（波は薄れ、感度は上がる）。",fontsize=8.2,color="#555",va="center")
arr((45,54.6),(45,64.6),C["ink"]); a.text(44,56.3,"内面の向き",fontsize=8.2,va="center",ha="right",color=C["ink"]); tag(43,59.4,"C",C["half"])
arr((59,54.6),(59,64.6),C["warn"],ls=":",lw=1.8); tag(57,59.4,"F",C["warn"]); a.text(52.6,56.6,"中身へは無い",fontsize=7.8,va="center",color=C["warn"],ha="center")
arr((66,64.6),(66,54.6),C["out"]); a.text(67,55.9,"声・読んだ語",fontsize=8.2,va="center",color=C["out"]); tag(68.2,58.6,"A",C["ok"]); tag(71.8,58.6,"B",C["ok"])
box(2,30,21,24,fc="#e8f1f3",ec=C["read"])
a.text(12.5,51.4,"拾われやすさ",ha="center",fontsize=10.5,color=C["read"],va="center")
a.text(12.5,41,"場の番を聞いて数える\n「前の番の語が、次の番で\n拾われたか」\n\n数は時間で薄れる\n話し手の組ごとに持つ",ha="center",va="center",fontsize=8.3)
arr((14,54.6),(29.6,66.5),C["read"],rad=-.15,ls="--",lw=1.2); a.text(15.5,61,"選ぶ重さ",fontsize=8.2,color=C["read"],ha="center")
# 違和感の波
box(83,30,35,24,fc="#eef6f0",ec=C["odd"],ls="--")
a.text(100.5,51.4,"違和感の波",ha="center",fontsize=10.5,color=C["odd"],va="center")
a.text(100.5,42.6,"種類ごとに 1 本の向き、強さは振幅、\n時間で薄れる（速さは器と同じ）\n\n材料が不足・途中・もう言った\n迷い・自信のなさ・組の珍しさ・展開②",ha="center",va="center",fontsize=8.4)
a.text(100.5,32.4,"器の感度にだけつないだ（骨組み）。全部記録する。",ha="center",fontsize=8.2,color="#555",va="center")
arr((84,64.6),(96,54.6),C["odd"],ls="--",lw=1.2); a.text(84.5,58.4,"毎歩の\n違和感",fontsize=8.2,color=C["odd"],ha="right",va="center")
arr((82.6,42),(77.5,42),C["half"],ls="--",lw=1.8); tag(80,44.6,"D",C["half"])
# 番号
num(88.6,92,1); num(89.5,112.5,2); num(117.5,115.5,3); num(31,91.5,4); num(31,53.5,5); num(117.5,53.5,6); num(22.5,53.5,7); num(117.5,88.5,8)
# 新しい段: 数えの部品・介入・当たり外れ
box(20,8,42,16,fc="#f4f0e6",ec=C["ink"])
a.text(41,21.4,"数えの部品（器を見ずに動いている）",ha="center",fontsize=10,va="center")
a.text(41,14.2,"関係の表: 3 つ組を数え、2 つから残りの 1 つを答える\n空席に座らせる・下がる順を数える\n算数: ずれを運ぶ・席の入れ替え・印ごとの事実",ha="center",va="center",fontsize=8.3,linespacing=1.5)
box(66,8,18,16,"介入\n\n気づいた後に\n直す働き",fc="#fbeeee",ec=C["warn"],fs=8.8,ls=":")
box(88,8,30,16,"当たり外れ\n\n正解・不正解の返事\n（算数では、数えの部品にだけ返した）",fc="#fbeeee",ec=C["warn"],fs=8.6,ls=":")
arr((41,24.6),(41,29.5),C["warn"],ls=":",lw=1.8); tag(43.6,27,"H",C["warn"])
arr((73,24.6),(70,29.5),C["warn"],ls=":",lw=1.8); tag(75,27,"E",C["warn"])
arr((93,24.6),(76.5,29.6),C["warn"],ls=":",lw=1.8); tag(87,28.2,"I",C["warn"])
a.plot([25,25],[24.6,58],color=C["warn"],lw=1.8,ls=":"); arr((25,57.5),(27.2,63.2),C["warn"],ls=":",lw=1.8); tag(25,27.4,"G",C["warn"])
# 問題点
a.plot([2,118],[3.6,3.6],color=C["line"],lw=1)
a.text(2,1.2,"問題点",fontsize=11.5,color=C["warn"],va="center")
P=["「材料が不足」の測り方は仮の物。内面は聞いた語そのものなので、聞いた語が入れば不足はすぐ消える。",
   "連想は本物の語では聞いていない語を連れて来る（83%）が、行が長くなるだけで、良くなったかは測れていない。",
   "組立に「並べる」段が無い（聞いた順のまま）。行づくりの違和感は、ほぼ行の長さで決まり、選びの良し悪しは測れない。",
   "話すきっかけが、外から付けた印（自分の番・文の終わり）だけ。内面の状態から話し始める事は、まだ無い。",
   "器の状態が、話す中身を変えていない。違和感を器に戻しても、60 ページで話した 58 行は全部同じだった。",
   "違和感は強さだけで、向きを持たない。どちらへ直せばよいかを器に伝えられない。",
   "拾われやすさで選んでも、最初の 3 語を言うのと差が出なかった。",
   "聞き返し役への言い直しは、近さに切る線が無く、「違う」と断れない。",
   "関係の表の当たりは 1 位で 4 割台。算数で効いた見方は、どれも手で置いた物で、自分で見つけてはいない。"]
for i,t in enumerate(P):
    y=-1.8-i*2.45; num(3.6,y,i+1 if i<8 else 9); a.text(6.4,y,t,fontsize=9,va="center",color=C["ink"])
# 矢印の一覧
a.plot([2,118],[-24.4,-24.4],color=C["line"],lw=1)
a.text(2,-26.8,"矢印の一覧",fontsize=11.5,color=C["ink"],va="center")
for x,col,t in ((24,C["ok"],"繋がっている"),(42,C["half"],"仮・骨組みだけ"),(62,C["warn"],"無い")):
    tag(x,-26.8," ",col); a.text(x+2.4,-26.8,t,fontsize=9,va="center",color=col)
R=[("A","ok","入力 → 器","読んだ語を器が聞く。聞く前に言い当て、外れを測る。"),
   ("B","ok","自分の声 → 器","出した行を一度戻す。出どころを「自分」と記録する。"),
   ("C","half","器 → 連想","内面の向き 1 本で近い語を引く。仮の連想で、本物の属性の板とは繋がっていない。"),
   ("D","half","違和感 → 器","違和感がいつもより強いと感度を上げる。向きは動かさない。話す行は変わらなかった。"),
   ("E","warn","介入 → 器","直した事が器に残らない。まだ作っていない。"),
   ("F","warn","器 → 組立・出力の中身","感度がどこにも使われていない。ここが切れているので、器が何をしても話す行が同じ。"),
   ("G","warn","関係の表 → 思考のループ","表の答えを、入れ物の材料にする口が無い。3 つ組を取り出すのも外の道具。"),
   ("H","warn","数えの部品 → 器","数えの部品は器を見ずに動く。知らないと気づいた事（迷い）が器に届かない。"),
   ("I","warn","当たり外れ → 器","正解・不正解は数えにだけ返した。器は当たったか外れたかを知らない。"),
   ("J","warn","思考のループ → 会話","本物の会話にはまだ繋いでいない（自分から話すのを強制しない方針）。")]
for i,(ch,k,n,t) in enumerate(R):
    y=-30.6-i*2.6; tag(3.6,y,ch,C[k]); a.text(6.4,y,n,fontsize=9.2,va="center",color=C[k]); a.text(34,y,t,fontsize=9,va="center",color=C["ink"])
a.text(2,-57.6,"読み（数えて確かめた物ではない）: いちばん大きい切れ目は F。器へ入る矢印を増やしても、出る矢印が無いと振る舞いは変わらない。",fontsize=9,color="#555",va="center")
fig.savefig(sys.argv[1],dpi=170); plt.close(fig)
