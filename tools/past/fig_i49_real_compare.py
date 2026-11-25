"""I49 の図: 本物の文での比べ（記録 14〜18 節の数字を、そのまま描く）。
使い方: python tools/past/fig_i49_real_compare.py 出力.png
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import sys, os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FONT=os.environ.get("KOTOYUI_FONT","/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT); plt.rcParams["font.family"]=fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"]=False
INK="#30323d"; RED="#d1495b"; BLUE="#00798c"; GOLD="#c9861a"; GREY="#9a9a9a"; GREEN="#5b8e3e"
fig=plt.figure(figsize=(14,9.6)); fig.patch.set_facecolor("white")
fig.suptitle("本物の文での比べ（会話 600 本、8,229 問、1 位の当たり。記録 14〜18 節の数字）",fontsize=15,color=INK,y=0.975)
# ① 形ごと（まっさらから）
ax=fig.add_axes([0.07,0.57,0.50,0.32])
rows=[("関係の表",.432,GREY),("解く・属性\n席あり\n元の向き",.408,BLUE),("今の波・属性\n席あり\n遠ざける",.356,RED),("今の波・属性\n席あり",.348,RED),("行の語・属性\n席なし\n元の向き",.329,BLUE),("行の語・語の波\n席なし\n元の向き",.244,BLUE),("今の波・属性\n席なし",.222,RED)]
for i,(n,v,c) in enumerate(rows):
    ax.bar(i,v,0.7,color=c,alpha=.9); ax.text(i,v+0.008,("%.3f"%v)[1:],ha="center",fontsize=11,color=INK)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows],fontsize=9.5); ax.set_ylim(0,0.5); ax.set_ylabel("1 位の当たり",fontsize=11,color=INK)
ax.set_title("① 形ごとの当たり（まっさらから）。青 = 元の向きを足す、赤 = 今の波に足す",fontsize=12.5,color=INK,loc="left")
ax.annotate("",xy=(1,0.47),xytext=(4,0.47),arrowprops=dict(arrowstyle="<->",color=INK,lw=1.2)); ax.text(2.5,0.478,"席を見る・見ないの差 8 ポイント",ha="center",fontsize=10.5,color=INK)
for s in ("top","right"): ax.spines[s].set_visible(False)
# ② 先に聞かせる量
ax=fig.add_axes([0.65,0.57,0.31,0.32])
x=[0,1,2,3,4]; lab=["無し","64 行に\n1 行","16 行に\n1 行","4 行に\n1 行","全部\n27 万行"]
ax.plot(x,[.329,.278,.265,.251,.222],"-o",color=BLUE,lw=2.2); ax.text(4.08,.222,"行の語・属性",color=BLUE,fontsize=10.5,va="center")
ax.plot(x,[.244,.203,.188,.176,.150],"-o",color=GOLD,lw=2.2); ax.text(4.08,.150,"行の語・語の波",color=GOLD,fontsize=10.5,va="center")
ax.plot([0,2],[.222,.123],"-o",color=RED,lw=2.2); ax.text(2.1,.118,"今の波・属性",color=RED,fontsize=10.5,va="center")
for xi,m in zip(x,[0,6,17,65,256]): ax.text(xi,0.335 if xi else 0.345,"%d"%m,ha="center",fontsize=9.5,color=GREY)
ax.text(2,0.368,"灰色の数 = 1 語の波に乗った語の数（中央値）",ha="center",fontsize=9.5,color=GREY)
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=9.5); ax.set_xlim(-0.3,5.6); ax.set_ylim(0,0.40); ax.set_xlabel("先に聞かせた、青空文庫の本当の行（1 行平均 8 語）",fontsize=10.5,color=INK)
ax.set_title("② 本当の行を先に聞かせるほど下がる",fontsize=12.5,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
# ③ 同じ乗り方で材料だけ違う
ax=fig.add_axes([0.07,0.09,0.25,0.32])
ax.bar([0,1],[.314,.278],0.6,color=[BLUE,GOLD]); 
for i,v in enumerate([.314,.278]): ax.text(i,v+0.006,("%.3f"%v)[1:],ha="center",fontsize=11.5,color=INK)
ax.set_xticks([0,1]); ax.set_xticklabels(["3 つ組を先に\n（1 行 3 語）","本当の行を先に\n（1 行 8 語）"],fontsize=10); ax.set_ylim(0,0.40)
ax.set_title("③ 乗った数が同じ（中央値 6）でも\n   材料で 3.6 ポイント違う",fontsize=12,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
# ④ 抑える形
ax=fig.add_axes([0.40,0.09,0.25,0.32])
st=["まっさら","3 つ組","全部の行"]; plain=[.329,.314,.222]; sub=[.256,.262,.186]; light=[.308,.313,.225]
xx=np.arange(3)
for d,vals,c,l in ((-0.26,plain,BLUE,"そのまま"),(0,light,GREEN,"よく出る語を軽く"),(0.26,sub,RED,"たまたまの分を引く")):
    ax.bar(xx+d,vals,0.25,color=c,label=l)
ax.set_xticks(xx); ax.set_xticklabels(st,fontsize=10); ax.set_ylim(0,0.42); ax.legend(fontsize=9.5,frameon=False,loc="upper right",ncol=1)
ax.set_title("④ 乗りすぎを抑える形は\n   効かないか、下げた",fontsize=12,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
# ⑤ 席ごと
ax=fig.add_axes([0.73,0.09,0.24,0.32])
seats=["だれが","何を","どうした"]
for d,vals,c,l in ((-0.26,[.573,.309,.415],GREY,"関係の表"),(0,[.505,.307,.413],BLUE,"解く・属性（席あり）"),(0.26,[.346,.265,.374],GOLD,"行の語・属性（席なし）")):
    ax.bar(xx+d,vals,0.25,color=c,label=l)
ax.set_xticks(xx); ax.set_xticklabels(seats,fontsize=10.5); ax.set_ylim(0,0.75); ax.legend(fontsize=9.5,frameon=False,loc="upper right")
ax.set_title("⑤ 席ごと: 差が出るのは\n   「だれが」",fontsize=12,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.text(0.5,0.468,"下の 3 枚は「行の語・属性」（席を見ず、同じ行の語の元の向きを足す形）についての数字。",ha="center",fontsize=10.5,color=INK)
fig.text(0.5,0.018,"読み: 表との差は、足し方ではなく「だれと組むかを絞る事」の側にある。席を見ると 8 ポイント上がり、関係の薄い語が混ざる材料では下がり、乗りすぎを抑える工夫では戻らない。",ha="center",fontsize=11,color=INK)
fig.savefig(sys.argv[1],dpi=170)
