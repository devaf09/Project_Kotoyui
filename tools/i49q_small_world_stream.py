"""I49q: 本物の文に近づけた小さい自作の世界（乱数なし）で、語の波の作り方を、聞く前に当てる流れで比べる。

i49q_small_world.py の世界は、同じ文を何度もくり返し聞かせてから試す作りで、きれいすぎた（本物の文では結果が合わなかった。記録 18 節）。
ここでは本物の数えに寄せる: 会話 600 本 × 3 つ組 14 個を 1 回だけ流し、聞く前に 1 語を隠して当てる。候補はその会話に出る語。
語は何個かの話にまたがり、話の出方に偏りが有り、よく出る語（私・それ・する）が混ざる。見せた語そのものは答えにしない。
使い方: python tools/i49q_small_world_stream.py   （画面に出すだけ）
手で置く物: 世界の形（話 120 個、人 6・物 8・動き 5、またがる幅 3、よく出る語の割合 0.25）、話や語の選び方の式、位相の本数。
"""
import math, json, sys, time, numpy as np
G=0.6180339887498949; R2=0.4142135623730951
def frac(x): return x-math.floor(x)
def make(T=120,talks=600,per=14,hub=0.25,spread=3):
    """話 T 個（人 6・物 8・動き 5）。語は spread 個の話にまたがる。会話 1 本 = 2 つの話から per 個の 3 つ組。よく出る語（私・それ・する）を hub の割合で混ぜる。"""
    def topic(t):
        S=["人%d"%((t*2+d*(1+(d*t)%spread))%(T*2)) for d in range(6)]
        O=["物%d"%((t*3+d*(1+(d+t)%spread))%(T*3)) for d in range(8)]
        V=["動%d"%((t*2+d*(1+(d*d+t)%spread))%(T*2)) for d in range(5)]
        return S,O,V
    tops=[topic(t) for t in range(T)]
    out=[]; q=0
    for n in range(talks):
        ta=int(frac((n+1)*G)**2*T)                      # 話の出方に偏り（2 乗でよく出る話が出来る）
        tb=(ta+1+int(frac((n+1)*R2)*5))%T
        tris=[]
        for j in range(per):
            q+=1; S,O,V=tops[ta if j%3 else tb]
            s=S[int(frac(q*G)**2*len(S))]; o=O[int(frac(q*R2)**2*len(O))]; v=V[int(frac(q*G*R2*7)**2*len(V))]
            if frac(q*0.7548776662)<hub: s="私"
            if frac(q*0.5698402910)<hub*0.6: v="する"
            if frac(q*0.3247179572)<hub*0.4: o="それ"
            tris.append((s,o,v))
        out.append((tris,sorted({w for t in tris for w in t})))
    return out
def base_dense(words,K):
    i=np.arange(1,len(words)+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
    return np.exp(2j*np.pi*((i*G*k*k+i*i*R2*k)%1.0))/math.sqrt(K)
class Table:
    """関係の表（比べる元）: 席の組ごとに、一緒に聞いた数を数える。聞いた事の無い候補は、元の向きの近さで補う事はしない。"""
    def __init__(s,n,B): s.c={}
    def ask(s,si,ids,cands):
        sc=np.zeros(len(cands))
        for sj in range(3):
            if sj!=si:
                for k,c in enumerate(cands): sc[k]+=math.log1p(s.c.get((si,sj,c,ids[sj]),0))
        for sj in range(3):
            if sj!=si: sc[cands==ids[sj]]=-1
        return sc
    def learn(s,ids):
        for si in range(3):
            for sj in range(3):
                if si!=sj: s.c[(si,sj,ids[si],ids[sj])]=s.c.get((si,sj,ids[si],ids[sj]),0)+1
class Vec:
    """live=False: 元の向きを足す（別の波に、相手の元の向きを足す）。live=True: 今の波に足す。seat=True: 席の組ごとに位相を回す。far: ほかを遠ざける。"""
    def __init__(s,n,B,live=False,seat=True,far=False):
        s.B=B; s.live=live; s.seat=seat; s.far=far; K=B.shape[1]
        s.W=B.copy() if live else np.zeros_like(B)
        Rs=np.exp(2j*np.pi*((np.arange(3)[:,None]+1)*G*(np.arange(K)[None,:]+1)*(1 if seat else 0)))
        s.rel=[[Rs[a]*np.conj(Rs[b]) for b in range(3)] for a in range(3)]
        s.sum=None
    def _u(s,a):
        v=s.W[a] if s.live else s.B[a]; n=np.linalg.norm(v); return v/n if n>0 else v
    def ask(s,si,ids,cands):
        aim=sum(s._u(ids[sj])*s.rel[si][sj] for sj in range(3) if sj!=si)
        C=s.W[cands]; n=np.linalg.norm(C,axis=1); n[n==0]=1
        sc=np.real(C@np.conj(aim))/n
        for sj in range(3):
            if sj!=si: sc[cands==ids[sj]]=-1
        return sc
    def learn(s,ids):
        u=[s._u(a) for a in ids]
        add=[sum(u[sj]*s.rel[si][sj] for sj in range(3) if sj!=si) for si in range(3)]
        if s.far:
            if s.sum is None: s.sum=np.zeros(s.B.shape[1],dtype=complex); s.cnt=0; s.seen={}
            m=s.sum/max(1,s.cnt)
            add=[x-2*m for x in add]
        for a,x in zip(ids,add):
            if s.far:
                old=s.seen.get(a)
            s.W[a]=s.W[a]+x
            if s.far:
                new=s.W[a]/max(np.linalg.norm(s.W[a]),1e-30)
                if old is None: s.cnt+=1; s.sum=s.sum+new
                else: s.sum=s.sum+new-old
                s.seen[a]=new
def stream(model,talks,IDX):
    hit=tot=0; per=[[0,0] for _ in range(3)]; parts=[]
    for n,(tris,ws) in enumerate(talks):
        cands=np.array([IDX[w] for w in ws])
        for t in tris:
            ids=[IDX[w] for w in t]
            for si in range(3):
                sc=model.ask(si,ids,cands)
                ok=sc.max()>0 and cands[int(np.argmax(sc))]==ids[si]
                hit+=ok; tot+=1; per[si][0]+=ok; per[si][1]+=1
            model.learn(ids)
        if (n+1)%100==0: parts.append(round(hit/tot,3))
    return round(hit/tot,3),[round(a/b,3) for a,b in per],parts
FORMS=[("関係の表",lambda n,B:Table(n,B)),("元の向きを足す・席あり",lambda n,B:Vec(n,B,False,True)),("元の向きを足す・席なし",lambda n,B:Vec(n,B,False,False)),
       ("今の波に足す・席あり",lambda n,B:Vec(n,B,True,True)),("今の波に足す・席あり・遠ざける",lambda n,B:Vec(n,B,True,True,True)),
       ("今の波に足す・席なし",lambda n,B:Vec(n,B,True,False)),("今の波に足す・席なし・遠ざける",lambda n,B:Vec(n,B,True,False,True))]
if __name__=="__main__":
    talks=make(); words=sorted({w for _,ws in talks for w in ws}); IDX={w:i for i,w in enumerate(words)}
    print("語",len(words),"会話",len(talks))
    for K in (64,256,1024):
        B=base_dense(words,K); print("本数",K)
        for name,mk in FORMS:
            a,per,parts=stream(mk(len(words),B),talks,IDX); print("  %-22s"%name,a,[float(x) for x in per],[float(x) for x in parts])
