"""I49b の記録の切り出し（数の列の JSON）から、反応の波の図を 2 枚描く。本文は扱わない（数だけ）。

図 1: 文章に合わせて動く波と動かない波（間なしの走りの 300 文。外れ・大きさ・感度、波 4 本）
図 2: 時間がたつだけで変わる波（黙る刻みを差し込んだ走りの 300 刻み。灰色が黙っている間）
JSON の形: seg1 / seg2 それぞれに sur・act・gain（波の番号 → 並び）、src（0 = tick、1 = 聞いた）、pin、seg1 は marks。
並びは、そのままの数の並びでも、[値, 続く数] の組の並びでもよい。
字の形は環境変数 KOTOYUI_FONT（日本語の出る字のファイル）で渡す。

python tools/i49b_wave_plot.py <数の列.json> <図 1.png> <図 2.png>
"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

FONT = os.environ.get("KOTOYUI_FONT")
if FONT:
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":11,"axes.titlelocation":"left","font.size":9.5})


def expand(v):
    return [x for x, n in v for _ in range(n)] if v and isinstance(v[0], list) else v


d = json.load(open(sys.argv[1], encoding="utf-8"))
for seg in (d["seg1"], d["seg2"]):
    seg["src"] = expand(seg["src"])
    for q in ("sur", "act", "gain"):
        seg[q] = {k: expand(v) for k, v in seg[q].items()}

B=["0","6","12","19"]; NM=["半減期 1（いちばん速い）","半減期 9","半減期 80","半減期 1,024（いちばん遅い）"]
C=["#d1495b","#edae49","#00798c","#30323d"]
def runs(src):
    r=[];i=0
    while i<len(src):
        if src[i]==0:
            j=i
            while j<len(src) and src[j]==0: j+=1
            r.append((i,j)); i=j
        else: i+=1
    return r
# ---- fig 1
s=d["seg1"]; x=np.arange(300)
fig,ax=plt.subplots(3,1,figsize=(10,8.2),sharex=True)
fig.suptitle("図 1  文章に合わせて動く波と、動かない波（本物の文 300 文、間なし）",x=.01,ha="left",fontsize=12.5)
tk=np.array(s["src"])==0
a=ax[0]; a.set_title("外れ（驚き）: 1 文ごとに大きく揺れる。速い波も遅い波もほぼ同じ形")
for k in (0,3):
    y=np.array(s["sur"][B[k]],float); y[tk]=np.nan; a.plot(x,y,color=C[k],lw=1.1,label=NM[k])
a.set_ylim(0,1.05); a.set_ylabel("外れ"); a.legend(frameon=False,ncol=2,loc="lower left",fontsize=8.5)
a=ax[1]; a.set_title("波の大きさ: 速い波は文ごとに動き、遅い波はほとんど動かない")
for k in range(4): a.plot(x,s["act"][B[k]],color=C[k],lw=1.2,label=NM[k])
a.set_ylabel("大きさ（上限 1）"); a.set_ylim(.2,.8); a.legend(frameon=False,ncol=4,loc="upper right",fontsize=8.5)
a=ax[2]; a.set_title("感度: 速い波は外れに合わせて上下し、遅い波は文に関係なくゆっくり上がるだけ")
for k in range(4): a.plot(x,s["gain"][B[k]],color=C[k],lw=1.2)
a.set_ylabel("感度"); a.set_xlabel("文の番号")
for a in ax:
    a.axvline(150,color="#555",lw=1.2,ls="-",zorder=0)
    for h in s["marks"]["見出し"]: a.axvline(h,color="#999",lw=1,ls="--",zorder=0)
ax[1].text(152,.215,"作品の替わり目（直後に見出し）",ha="left",fontsize=8.5,color="#333"); ax[1].text(94,.215,"見出し",ha="right",fontsize=8.5,color="#666")
fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(sys.argv[2],dpi=200); plt.close(fig)
# ---- fig 2
s=d["seg2"]
fig,ax=plt.subplots(3,1,figsize=(10,8.2),sharex=True)
fig.suptitle("図 2  時間がたつだけで変わる波（黙る刻みを 100 ずつ差し込んだ走り、300 刻み）",x=.01,ha="left",fontsize=12.5)
tk=np.array(s["src"])==0
a=ax[0]; a.set_title("外れ: 聞いている間だけ測れる（黙っている間は線なし）。黙った直後の速い波は 1.0（何も言い当てられない）")
for k in (0,3):
    y=np.array(s["sur"][B[k]],float); y[tk]=np.nan; a.plot(x,y,color=C[k],lw=1.3,marker="o",ms=2.5,label=NM[k])
a.set_ylim(0,1.08); a.set_ylabel("外れ"); a.legend(frameon=False,ncol=2,loc="lower center",fontsize=8.5)
a=ax[1]; a.set_title("波の大きさ: 黙ると速い波から消える。聞くと速い 2 本は上限に張り付く（太い印）")
for k in range(4):
    y=np.array(s["act"][B[k]]); a.plot(x,y,color=C[k],lw=1.3,label=NM[k])
    p=s["pin"][B[k]]
    if p: a.plot(p,y[p]+(.03 if k==0 else .06),ls="",marker="s",ms=3,color=C[k])
a.set_ylim(0,1.12); a.set_ylabel("大きさ（上限 1）")
a=ax[2]; a.set_title("感度: 黙っている間に上がり続け、速い波は上限 16 に着く。聞くと下がるが、遅めの波は戻りきらない")
for k in range(4): a.plot(x,s["gain"][B[k]],color=C[k],lw=1.3,label=NM[k])
a.legend(frameon=False,ncol=2,loc="lower left",bbox_to_anchor=(.2,0),fontsize=8.5)
a.set_yscale("log",base=2); a.set_yticks([.5,1,2,4,8,16]); a.set_yticklabels(["0.5","1","2","4","8","16"]); a.axhline(16,color="#bbb",lw=.8,ls=":"); a.set_ylabel("感度（上限 16）"); a.set_xlabel("刻み（灰色 = 黙っている間、白 = 文を聞いている間）")
for a in ax:
    for (i,j) in runs(s["src"]):
        if j-i>3: a.axvspan(i-.5,j-.5,color="#ececec",zorder=0,lw=0)
fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(sys.argv[3],dpi=200); plt.close(fig)
