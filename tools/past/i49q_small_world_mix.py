"""I49q: 小さい自作の世界の 1 行に、関係の薄い語とよく出る語を混ぜて、語の波の作り方を総当たりで比べる（乱数なし。席は見ない＝行の語だけ）。

世界は tools/past/i49q_small_world.py と同じ。1 行 = 3 つ組 + 混ぜた語（半分はよく出る 5 語、半分は行の番号から決まる遠い語）。
比べる物: 形 7 つ × 位相の本数 × 混ぜた語の数。
  今の波に足す ／ ほかを遠ざける ／ 元の波を足す ／ たまたまの分を引く ／
  周波数を寄せる（語ごとに本数ぶんの周波数を数として持ち、同じ行の語の平均へ、聞いた回数で割った分だけ動かす。持ち主の案「周波数を少し変える」、未確定）／
  ほかから離す ／ 元の周波数の平均
使い方: python tools/past/i49q_small_world_mix.py 出力.json   （記録はリポジトリの外へ）
手で置く物: 混ぜる語の選び方、くり返し 5 回、周波数を寄せる時の動かす幅、周波数の近さを差の大きさの平均で測る事。
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import json, sys, time, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i49q_small_world import world, G, R2  # noqa: E402
FORMS=["今の波に足す","今の波に足す・ほかを遠ざける","元の波を足す","元の波を足す・たまたまの分を引く","周波数を寄せる","周波数を寄せる・ほかから離す","元の周波数の平均"]
def run(T,Ks,Ms,passes=5):
    heard,plaus,odd,half,ok,SEAT_W=world(T)
    COMMON=["よく出る%d"%i for i in range(5)]
    WORDS=list(dict.fromkeys([w for s in SEAT_W for w in s]+COMMON)); IDX={w:i for i,w in enumerate(WORDS)}; n=len(WORDS); nc=n-5
    rows=[]
    for M in Ms:
        H=[]
        for L,t in enumerate(heard):
            ids=[IDX[w] for w in t]
            for j in range(M):
                d=IDX[COMMON[(L+j)%5]] if j%2==0 else (L*7919+j*104729+13)%nc
                if d not in ids: ids.append(d)
            H.append(np.array(ids))
        for K in Ks:
            i=np.arange(1,n+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
            ph=(i*G*k*k+i*i*R2*k)%1.0
            B=np.exp(2j*np.pi*ph)
            for form in FORMS:
                freq=form.endswith("平均") or form.startswith("周波数")
                cur=form.startswith("今") or form.startswith("周波数")
                far="遠ざける" in form or "離す" in form
                X=ph.copy() if freq else B.copy()                       # 周波数の形: 語ごとに K 本の周波数（0〜1 の数）を持つ
                W=X.copy() if cur else np.zeros_like(X); cnt=np.zeros(n); mass=np.zeros(n)
                for _ in range(passes if cur else 1):
                    for ids in H:
                        m=len(ids)
                        if freq:
                            src=W[ids] if cur else X[ids]
                            mates=(src.sum(0)[None,:]-src)/(m-1)        # 同じ行のほかの語の周波数の平均
                            if cur:
                                c=cnt[ids][:,None]
                                step=(mates-W[ids])/(c+1)               # 聞いた回数で割って寄せる（回数が増えるほど小さく動く）
                                if far: step=step-(W.mean(0)[None,:]-W[ids])/(c+1)
                                W[ids]=W[ids]+step
                            else:
                                W[ids]=W[ids]+mates
                            cnt[ids]+=1
                        else:
                            u=W[ids]/np.linalg.norm(W[ids],axis=1,keepdims=True) if cur else X[ids]
                            x=u.sum(0)[None,:]-u
                            if far: x=x-(m-1)*(W/np.linalg.norm(W,axis=1,keepdims=True)).mean(0)[None,:]
                            W[ids]=W[ids]+x; mass[ids]+=m-1
                if form=="元の周波数の平均": W=W/np.maximum(cnt,1)[:,None]
                if form=="元の波を足す・たまたまの分を引く": W=W-mass[:,None]*(W.sum(0)/max(1,mass.sum()))
                if freq:
                    Tg=W if cur else X
                    S=-0.5*(np.abs(W[:,None,:]-Tg[None,:,:]).mean(-1)+np.abs(Tg[:,None,:]-W[None,:,:]).mean(-1)) if n*n*K<3e8 else None
                    if S is None:
                        S=np.zeros((n,n))
                        for a in range(n): S[a]=-np.abs(W[a][None,:]-Tg).mean(-1)
                        S=0.5*(S+S.T)
                else:
                    Tg=W if cur else X
                    Wn=W/np.maximum(np.linalg.norm(W,axis=1,keepdims=True),1e-30); Tn=Tg/np.linalg.norm(Tg,axis=1,keepdims=True)
                    S=np.real(Wn@np.conj(Tn).T); S=0.5*(S+S.T)
                def sc(t):
                    a,b,c=(IDX[w] for w in t); return S[a,b]+S[a,c]+S[b,c]
                auc=lambda hi,lo: float(np.mean(np.array(hi)[:,None]>np.array(lo)[None,:]))
                hit=tot=0
                for t in plaus[::max(1,len(plaus)//50)]:
                    for si in range(3):
                        best=max(SEAT_W[si],key=lambda x:sc(tuple(x if j==si else t[j] for j in range(3))))
                        hit+=tuple(best if j==si else t[j] for j in range(3)) in ok; tot+=1
                pl=[sc(t) for t in plaus]
                rows.append(dict(話=T,語=n,混ぜた語=M,形=form,本数=K,当て=round(hit/tot,3),変との区別=round(auc(pl,[sc(t) for t in odd]),3),
                                 半分変との区別=round(auc(pl,[sc(t) for t in half]),3)))
        print(T,M,flush=True)
    return rows
if __name__=="__main__":
    out=[]
    for T in (40,160):
        t0=time.time(); out+=run(T,(4,16,64,256,1024),(0,2,5,10)); print(T,round(time.time()-t0),flush=True)
        json.dump(out,open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False)
    for T in (40,160):
        print("\n=== 話",T)
        for M in (0,2,5,10):
            print(" 混ぜた語",M)
            for f in FORMS:
                print("   %-18s"%f[:18]," ".join("%.2f"%r["当て"] for r in out if r["話"]==T and r["混ぜた語"]==M and r["形"]==f),
                      " | 半分変:"," ".join("%.2f"%r["半分変との区別"] for r in out if r["話"]==T and r["混ぜた語"]==M and r["形"]==f))
