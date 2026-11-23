"""I49q: 小さい自作の世界で、混ぜ物を見分けて足す形（今の共振が強い相手ほど多く足す）を比べる。乱数なし、席なし。

使い方: python tools/i49q_small_world_gate.py 出力.json   （記録はリポジトリの外へ）
手で置く物: 重さの形（共振の正の分 + 1 ÷ (1 + 聞いた回数)、など 4 通り）、くり返し 5 回。
"""
import json, sys, time, numpy as np
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from i49q_small_world import world, G, R2
from i49q_small_world_attr import unitrows, lines_of, evaluate
def run(T=160,Ms=(0,5,10),passes=5):
    heard,plaus,odd,half,ok,SEAT_W=world(T)
    COMMON=["よく出る%d"%i for i in range(5)]
    WORDS=list(dict.fromkeys([w for s in SEAT_W for w in s]+COMMON)); IDX={w:i for i,w in enumerate(WORDS)}; n=len(WORDS); nc=n-5
    seat_of={IDX[w]:si for si in range(3) for w in SEAT_W[si]}
    def attr_base(g):
        A=np.zeros((n,T//g+3),dtype=np.complex128)
        for t in range(T):
            for w in ["私"]+["人%d"%((t+d)%T) for d in range(3)]+["物%d"%((2*t+d)%(2*T)) for d in range(3)]+["する"]+["動%d"%(2*t+d) for d in range(2)]: A[IDX[w],t//g]+=1
        for i,si in seat_of.items(): A[i,T//g+si]=1
        for c in COMMON: A[IDX[c],:T//g]=1
        return A
    def dense(K):
        i=np.arange(1,n+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
        return np.exp(2j*np.pi*((i*G*k*k+i*i*R2*k)%1.0))
    rows=[]
    for M in Ms:
        H=lines_of(heard,IDX,COMMON,nc,M)
        def live(W0,gate):
            W=W0.copy(); cnt=np.zeros(n)
            for p in range(passes):
                for ids in H:
                    u=unitrows(W[ids]); m=len(ids)
                    if gate=="そのまま":
                        x=u.sum(0)[None,:]-u
                    else:
                        C=np.real(u@np.conj(u).T); np.fill_diagonal(C,0.0)             # 行の中の語どうしの、今の共振
                        if gate=="共振の強い相手ほど多く":
                            Wt=np.maximum(C,0)+1.0/(1.0+cnt[ids])[:,None]
                        elif gate=="共振の向きごと（合わない相手は引く）":
                            Wt=C+1.0/(1.0+cnt[ids])[:,None]
                        elif gate=="1 回目はそのまま、2 回目からは共振の強い相手だけ":
                            Wt=np.ones((m,m)) if p==0 else np.maximum(C,0)
                        elif gate=="行の中で平均より強く共振する相手だけ":
                            Wt=(C>C.sum(1,keepdims=True)/(m-1)).astype(float) if p>0 else np.ones((m,m))
                        np.fill_diagonal(Wt,0.0); x=Wt@u
                    W[ids]=W[ids]+x; cnt[ids]+=1
            return W
        for bname,W0 in (("まんべんなく 64 本",dense(64)),("まんべんなく 256 本",dense(256)),("属性 43 本",attr_base(4)),("属性 13 本",attr_base(16))):
            for gate in ("そのまま","共振の強い相手ほど多く","共振の向きごと（合わない相手は引く）","1 回目はそのまま、2 回目からは共振の強い相手だけ","行の中で平均より強く共振する相手だけ"):
                t0=time.time(); a,hf=evaluate(live(W0,gate),IDX,SEAT_W,plaus,half,ok)
                rows.append(dict(混ぜた語=M,元=bname,足し方=gate,当て=a,半分変との区別=hf)); print(M,bname,gate,a,hf,round(time.time()-t0),flush=True)
    return rows
if __name__=="__main__":
    json.dump(run(),open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False)
