"""I49q の図: 関係の薄い語が混ざると、なぜ当たらなくなるか（小さい自作の世界）。
使い方: python tools/fig_i49q_why_mixing.py 総当たり2の記録.json 出力.png   （総当たり 2 の記録は tools/i49q_small_world_mix.py の出力）
"""
import json, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from i49q_small_world import world, G, R2
FONT=os.environ.get("KOTOYUI_FONT","/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf")
if os.path.exists(FONT):
    fm.fontManager.addfont(FONT); plt.rcParams["font.family"]=fm.FontProperties(fname=FONT).get_name(); plt.rcParams["axes.unicode_minus"]=False
INK="#30323d"; RED="#d1495b"; BLUE="#00798c"; GOLD="#c9861a"; GREY="#9a9a9a"; GREEN="#5b8e3e"
T=160; K=256
heard,plaus,odd,half,ok,SEAT_W=world(T)
COMMON=["よく出る%d"%i for i in range(5)]
WORDS=list(dict.fromkeys([w for s in SEAT_W for w in s]+COMMON)); IDX={w:i for i,w in enumerate(WORDS)}; n=len(WORDS); nc=n-5
i=np.arange(1,n+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
B=np.exp(2j*np.pi*((i*G*k*k+i*i*R2*k)%1.0))
def build(M):
    W=np.zeros_like(B); cnt_true=np.zeros((n,n)); cnt_noise=np.zeros((n,n))
    for L,t in enumerate(heard):
        ids=[IDX[w] for w in t]; tr=list(ids)
        for j in range(M):
            d=IDX[COMMON[(L+j)%5]] if j%2==0 else (L*7919+j*104729+13)%nc
            if d not in ids: ids.append(d)
        s=B[ids].sum(0)
        for a in ids:
            W[a]+=s-B[a]
            for b in ids:
                if a!=b: (cnt_true if (a in tr and b in tr) else cnt_noise)[a,b]+=1
    Wn=W/np.maximum(np.linalg.norm(W,axis=1,keepdims=True),1e-30); Tn=B/np.linalg.norm(B,axis=1,keepdims=True)
    S=np.real(Wn@np.conj(Tn).T); S=0.5*(S+S.T)
    right=[];wrong=[]
    for t in plaus[::3]:
        for si in range(3):
            for c in SEAT_W[si]:
                tt=tuple(c if j==si else t[j] for j in range(3)); a,b,cc=(IDX[w] for w in tt)
                (right if tt in ok else wrong).append(S[a,b]+S[a,cc]+S[b,cc])
    content=[IDX[w] for s in SEAT_W for w in s]
    et=(cnt_true[content]**2).sum(); en=(cnt_noise[content]**2).sum()
    return np.array(right),np.array(wrong),et/(et+en),cnt_true[content].sum()/(cnt_true[content].sum()+cnt_noise[content].sum())
res={M:build(M) for M in (0,2,5,10)}
g3=json.load(open(sys.argv[1],encoding="utf-8"))
fig=plt.figure(figsize=(14,10)); fig.patch.set_facecolor("white")
fig.suptitle("関係の薄い語が混ざると、なぜ当たらなくなるか（自作の小さい世界、802 語、位相 256 本）",fontsize=15,color=INK,y=0.975)
ax=fig.add_axes([0.06,0.56,0.40,0.33])
for f,c,l in (("今の波に足す",RED,"今の波に足す"),("元の波を足す",BLUE,"元の波を足す（今の本体の形）"),("元の波を足す・たまたまの分を引く",GREEN,"元の波 + たまたまの分を引く"),("周波数を寄せる",GOLD,"周波数を寄せる")):
    ys=[next(r["当て"] for r in g3 if r["話"]==160 and r["混ぜた語"]==M and r["形"]==f and r["本数"]==256) for M in (0,2,5,10)]
    ax.plot([0,2,5,10],ys,"-o",color=c,lw=2.2); ax.text(10.2,{"元の波を足す":0.085,"元の波を足す・たまたまの分を引く":0.025}.get(f,ys[-1]),l,color=c,fontsize=10.5,va="center")
ax.axhline(0.02,color=GREY,lw=1,ls=":"); ax.text(0.1,-0.045,"点線 = まぐれ",color=GREY,fontsize=9.5)
ax.set_xlim(-0.3,15.5); ax.set_ylim(-0.07,1.05); ax.set_xticks([0,2,5,10]); ax.set_xlabel("1 行に混ぜた、関係の薄い語の数",fontsize=11,color=INK); ax.set_ylabel("当てた割合",fontsize=11,color=INK)
ax.set_title("① 混ぜ物が増えると、どの形も下がる",fontsize=12.5,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax=fig.add_axes([0.58,0.56,0.38,0.33])
Ms=[0,2,5,10]; x=np.arange(4)
ax.bar(x-0.2,[2/(2+M) for M in Ms],0.38,color=GREY,label="足した回数のうち、本当の相手の割合")
ax.bar(x+0.2,[res[M][2] for M in Ms],0.38,color=BLUE,label="波の強さのうち、本当の相手の分の割合")
for j,M in enumerate(Ms):
    ax.text(j-0.2,2/(2+M)+0.02,"%.0f%%"%(200/(2+M)),ha="center",fontsize=10,color=INK); ax.text(j+0.2,res[M][2]+0.02,"%.0f%%"%(100*res[M][2]),ha="center",fontsize=10,color=INK)
ax.set_xticks(x); ax.set_xticklabels(["混ぜ物 %d 語"%M for M in Ms],fontsize=10.5); ax.set_ylim(0,1.2); ax.legend(fontsize=10,frameon=False,loc="upper right")
ax.set_title("② 語の波の中身: 本当の相手の分が、混ぜ物に埋もれていく",fontsize=12.5,color=INK,loc="left")
for s in ("top","right"): ax.spines[s].set_visible(False)
for j,M in enumerate((0,10)):
    ax=fig.add_axes([0.06+0.52*j,0.115,0.40,0.31]); r,w,_,_=res[M]
    lo=min(r.min(),w.min()); hi=max(r.max(),w.max()); bins=np.linspace(lo,hi,50)
    ax.hist(w,bins=bins,color=GREY,alpha=.75,density=True,label="合わない語（外れ）"); ax.hist(r,bins=bins,color=RED,alpha=.75,density=True,label="話に合う語（当たり）")
    sep=(r.mean()-w.mean())/w.std()
    ax.set_title("③ 混ぜ物 %d 語（山の離れ方 = 外れの散らばりの %.1f 倍）"%(M,sep),fontsize=12.5,color=INK,loc="left")
    ax.set_xlabel("候補の点（見せた 2 語とのそろい方）",fontsize=11,color=INK); ax.set_yticks([]); ax.legend(fontsize=10,frameon=False,loc="upper right")
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
fig.text(0.5,0.485,"下の 2 枚（元の波を足す形）: 赤い山（当たりの候補）と灰色の山（外れの候補）が離れているほど、当てやすい。",ha="center",fontsize=11,color=INK)
fig.text(0.5,0.463,"混ぜ物が 10 語あると 2 つの山が重なる。外れの候補は 300 語以上あるので、少し重なるだけで 1 位は外れになる。",ha="center",fontsize=11,color=INK)
fig.text(0.5,0.035,"読み: 1 行に本当の相手は 2 語だけ。混ぜ物が 10 語あると、足した回数の 8 割以上が混ぜ物になる。",ha="center",fontsize=11.5,color=INK)
fig.text(0.5,0.012,"同じ相手と何度も会わない限り、波の中で本当の相手は目立てない。位相の本数を増やしても、この割合は変わらない。",ha="center",fontsize=11.5,color=INK)
out=sys.argv[2]; fig.savefig(out,dpi=170); print(out,{M:(round(float(res[M][2]),3),round(float((res[M][0].mean()-res[M][1].mean())/res[M][1].std()),2)) for M in res})
