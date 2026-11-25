"""I49q: 小さい自作の世界で、属性の数だけ周波数を持つ形と、外れた時だけ周波数を足す形を比べる（持ち主の案、未確定。乱数なし、席なし、今の波に足す）。

  属性: 話をいくつかまとめた物を属性 1 つにし、語は持っている属性の周波数にだけ強さを持つ（席の種類も属性に入れる）。聞かせない時と、今の波に足した時を比べる。
  外れた時に足す: 初めは少ない本数。聞く前に当てて、外れが当たりより多くなった語が居た行で、新しい周波数を 1 本足し、その行の語をその上でそろえる。
使い方: python tools/past/i49q_small_world_attr.py 出力.json   （記録はリポジトリの外へ）
手で置く物: 属性のまとめ方（話 1・4・16 個で 1 つ）、外れの決め方（同じ席の上位 5 位に入らなければ外れ）、足す本数の上限 2,048、くり返し 5 回。
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import json, sys, time, numpy as np
import os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from i49q_small_world import world, G, R2
def unitrows(W):
    n=np.linalg.norm(W,axis=1,keepdims=True); return W/np.where(n==0,1,n)
def lines_of(heard,IDX,COMMON,nc,M):
    H=[]
    for L,t in enumerate(heard):
        ids=[IDX[w] for w in t]
        for j in range(M):
            d=IDX[COMMON[(L+j)%5]] if j%2==0 else (L*7919+j*104729+13)%nc
            if d not in ids: ids.append(d)
        H.append(ids)
    return H
def evaluate(W,IDX,SEAT_W,plaus,half,ok):
    Wn=unitrows(W); S=np.real(Wn@np.conj(Wn).T)
    sc=lambda t:(lambda a,b,c:S[a,b]+S[a,c]+S[b,c])(*(IDX[w] for w in t))
    hit=tot=0
    for t in plaus[::max(1,len(plaus)//50)]:
        for si in range(3):
            best=max(SEAT_W[si],key=lambda x:sc(tuple(x if j==si else t[j] for j in range(3))))
            hit+=tuple(best if j==si else t[j] for j in range(3)) in ok; tot+=1
    pl=np.array([sc(t) for t in plaus]); hf=np.array([sc(t) for t in half])
    return round(hit/tot,3), round(float(np.mean(pl[:,None]>hf[None,:])),3)
def run(T=160,Ms=(0,5,10),passes=5):
    heard,plaus,odd,half,ok,SEAT_W=world(T)
    COMMON=["よく出る%d"%i for i in range(5)]
    WORDS=list(dict.fromkeys([w for s in SEAT_W for w in s]+COMMON)); IDX={w:i for i,w in enumerate(WORDS)}; n=len(WORDS); nc=n-5
    seat_of={IDX[w]:si for si in range(3) for w in SEAT_W[si]}
    seat_ids=[np.array([IDX[w] for w in SEAT_W[si]]) for si in range(3)]
    # 話ごとの語（属性を作る元）
    topic_words=[]
    for t in range(T):
        topic_words.append(["私"]+["人%d"%((t+d)%T) for d in range(3)]+["物%d"%((2*t+d)%(2*T)) for d in range(3)]+["する"]+["動%d"%(2*t+d) for d in range(2)])
    def attr_base(g):
        A=np.zeros((n,T//g+3),dtype=np.complex128)
        for t,ws in enumerate(topic_words):
            for w in ws: A[IDX[w],t//g]+=1
        for i,si in seat_of.items(): A[i,T//g+si]=1          # 席の種類（人・物・動き）も属性の 1 つ
        for c in COMMON: A[IDX[c],:T//g]=1
        return A
    def dense(K):
        i=np.arange(1,n+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
        return np.exp(2j*np.pi*((i*G*k*k+i*i*R2*k)%1.0))
    rows=[]
    for M in Ms:
        H=lines_of(heard,IDX,COMMON,nc,M)
        def live(W,p=passes):
            W=W.copy()
            for _ in range(p):
                for ids in H:
                    u=unitrows(W[ids]); W[ids]=W[ids]+(u.sum(0)[None,:]-u)
            return W
        def grow(K0,Kmax=2048,p=passes,top=5):
            W=np.zeros((n,Kmax),dtype=np.complex128); W[:,:K0]=dense(K0); used=K0; hits=np.zeros(n); miss=np.zeros(n); added=0; log=[]
            for _ in range(p):
                h=m=0
                for ids in H:
                    V=W[:,:used]; u=unitrows(V[ids]); tot=u.sum(0)
                    new=False
                    for pos in range(3):
                        a=ids[pos]; aim=tot-u[pos]; cand=seat_ids[seat_of[a]]
                        s=np.real(unitrows(V[cand])@np.conj(aim))
                        good=(s>s[np.where(cand==a)[0][0]]).sum()<top
                        if good: hits[a]+=1; h+=1
                        else:
                            miss[a]+=1; m+=1
                            if miss[a]>hits[a]: new=True
                    V[ids]=V[ids]+(tot[None,:]-u)
                    if new and used<Kmax:                         # 違和感が続く語が居た行: 新しい周波数を 1 本足し、この行の語をその上でそろえる
                        W[ids,used]=1.0; used+=1; added+=1
                        for a in ids[:3]: hits[a]=miss[a]=0
                log.append((h,m,used))
            return W[:,:used],added,log
        forms=[("まんべんなく 16 本・今の波に足す",lambda:(live(dense(16)),None)),("まんべんなく 64 本・今の波に足す",lambda:(live(dense(64)),None)),
               ("まんべんなく 256 本・今の波に足す",lambda:(live(dense(256)),None))]
        for g in (1,4,16):
            forms.append(("属性 %d 本（話 %d 個で 1 つ）・聞かない"%(T//g+3,g),lambda g=g:(attr_base(g),None)))
            forms.append(("属性 %d 本（話 %d 個で 1 つ）・今の波に足す"%(T//g+3,g),lambda g=g:(live(attr_base(g)),None)))
        for K0 in (4,16):
            forms.append(("初め %d 本・外れた時に足す"%K0,lambda K0=K0:grow(K0)[:2]))
        for name,f in forms:
            t0=time.time(); W,added=f(); a,hfv=evaluate(W,IDX,SEAT_W,plaus,half,ok)
            rows.append(dict(混ぜた語=M,形=name,本数=W.shape[1],足した本数=added,当て=a,半分変との区別=hfv))
            print(M,name,W.shape[1],added,a,hfv,round(time.time()-t0),flush=True)
    return rows
if __name__=="__main__":
    json.dump(run(),open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False)
