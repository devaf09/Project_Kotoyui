"""I49q: 小さい自作の世界で、語の波の作り方を総当たりで比べる（持ち主の案「小さい実験を総当たりでやってみよう」。乱数なし）。

世界: 話が T 個、輪になって隣の話と語を分け合う（話ごとに 人 4・物 3・動き 3。「私」「する」はどの話にも出る）。3 つ組の 3 分の 1 は聞かせない。
比べる物: 形 6 つ × 語 1 つの位相の本数 × 席の回し（なし・あり）× くり返し。
  今の波に足す（聞くたびに語の波そのものが寄る）／ 共通の分を引く ／ ほかを遠ざける（足した数だけ、全部の語の平均を引く）／
  元の波を足す（I49k〜n の形）／ たまたまの分を引く ／ 夜に数え直す（偏り）
数える物: 覚え（聞いた文が、変な文より上に来る割合）、広がり（聞いていないがありそうな文が、変な文より上）、半分変との区別（2 語は合うが 3 語では合わない文より上）、
          当て（ありそうな文の 1 語を隠し、同じ席の語全部から選んで、話に合う語を選べた割合）。
使い方: python tools/past/i49q_small_world.py 出力.json   （記録はリポジトリの外へ）
手で置く物: 世界の形、聞かせない割合、くり返しの回数、遠ざける強さ、初めの位相の決め方（語の番号と周波数の番号から）。
"""
import os as _o, sys as _s; _s.path[:0] = [_o.path.join(_o.path.dirname(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))), *_d) for _d in (("intervention", "past"), ("tools", "past"), ("intervention",), ("tools",))]  # 前の試しの置き場（past/）から、元の置き場の部品も読めるようにする
import itertools, json, math, sys, time, numpy as np
G=0.6180339887498949; R2=0.4142135623730951
def world(T):
    """話が T 個、輪になって隣の話と語を分け合う。どの話にも出る語（私・する）も有る。"""
    topics=[]
    for t in range(T):
        S=["私"]+["人%d"%((t+d)%T) for d in range(3)]
        O=["物%d"%((2*t+d)%(2*T)) for d in range(3)]
        V=["する"]+["動%d"%(2*t+d) for d in range(2)]
        topics.append((S,O,V))
    ok=set(); heard=[]; plaus=[]
    for S,O,V in topics:
        for i,s in enumerate(S):
            for j,o in enumerate(O):
                for k,v in enumerate(V):
                    t=(s,o,v)
                    if t in ok: continue
                    ok.add(t); (plaus if (i+j+k)%3==0 else heard).append(t)
    hs=set(heard); plaus=[t for t in plaus if t not in hs]
    seats=tuple(sorted({tp[i][j] for tp in topics for j in range(len(tp[i]))}) for i in range(3))
    sets=[tuple(set(p) for p in tp) for tp in topics]
    def n_fit(t): return max(sum(x in part for x,part in zip(t,tp)) for tp in sets)
    odd=[];half=[]
    step=max(1,(len(seats[0])*len(seats[1])*len(seats[2]))//4000)
    for i,t in enumerate(itertools.product(*seats)):
        if i%step or t in ok: continue
        f=n_fit(t)
        if f<=1: odd.append(t)
        elif "私" not in t and "する" not in t: half.append(t)
    cut=lambda L,n: L[::max(1,len(L)//n)][:n]
    return heard,cut(plaus,150),cut(odd,300),cut(half,300),ok,seats

def run(T,Ks,passes_list,forms):
    heard,plaus,odd,half,ok,SEAT_W=world(T)
    WORDS=list(dict.fromkeys(w for s in SEAT_W for w in s)); IDX={w:i for i,w in enumerate(WORDS)}; n=len(WORDS)
    H=[[IDX[w] for w in t] for t in heard]
    rows=[]
    for K in Ks:
        i=np.arange(1,n+1)[:,None].astype(float); k=np.arange(1,K+1)[None,:].astype(float)
        B=np.exp(2j*np.pi*((i*G*k*k+i*i*R2*k)%1.0))
        for seat in (False,True):
            Rs=np.exp(2j*np.pi*((np.arange(3)[:,None]+1)*G*(np.arange(K)[None,:]+1)*(1 if seat else 0)))
            REL=[[Rs[a]*np.conj(Rs[b]) for b in range(3)] for a in range(3)]
            for form in forms:
                cur=form.startswith("今")
                for p in (passes_list if cur else passes_list[:1]):
                    W=B.copy() if cur else np.zeros_like(B); mass=np.zeros(n); cnt={}; nw={}; nc={}; npair={}
                    far=form=="今の波に足す・ほかを遠ざける"
                    if far: U=W/np.linalg.norm(W,axis=1,keepdims=True); mean_all=U.mean(axis=0)
                    for _ in range(p):
                        for ids in H:
                            if cur: u=[W[a]/np.linalg.norm(W[a]) for a in ids]
                            else: u=[B[a] for a in ids]
                            add=[]
                            for si,a in enumerate(ids):
                                x=sum(u[sj]*REL[si][sj] for sj in range(3) if sj!=si)
                                if far: x=x-2*mean_all
                                add.append(x)
                                if not cur:
                                    for sj,b in enumerate(ids):
                                        if sj!=si:
                                            key=(a,si,sj,b); cnt[key]=cnt.get(key,0)+1; nw[(a,si)]=nw.get((a,si),0)+1
                                            nc[(si,sj,b)]=nc.get((si,sj,b),0)+1; npair[(si,sj)]=npair.get((si,sj),0)+1
                            for a,x in zip(ids,add):
                                if far: old=W[a]/np.linalg.norm(W[a])
                                W[a]=W[a]+x; mass[a]+=2
                                if far: mean_all=mean_all+(W[a]/np.linalg.norm(W[a])-old)/n
                    if form=="元の波を足す・夜に数え直す（偏り）":
                        W=np.zeros_like(B)
                        for (a,si,sj,b),c in cnt.items():
                            g=max(0.0,math.log(c*npair[(si,sj)]/(nw[(a,si)]*nc[(si,sj,b)])))
                            W[a]+=g*B[b]*REL[si][sj]
                    if form=="今の波に足す・共通の分を引く": W=W-W.mean(axis=0)
                    if form=="元の波を足す・たまたまの分を引く": W=W-mass[:,None]*(W.sum(axis=0)/max(1,mass.sum()))
                    Tg=W if cur else B
                    nW=np.linalg.norm(W,axis=1); nT=np.linalg.norm(Tg,axis=1)
                    def c(a,b,sx,sy):
                        d=nW[a]*nT[b]; return 0.0 if d==0 else float(np.real(np.vdot(W[a],Tg[b]*REL[sx][sy])))/d
                    def sc(t):
                        ids=[IDX[w] for w in t]
                        return float(np.mean([0.5*(c(ids[i],ids[j],i,j)+c(ids[j],ids[i],j,i)) for i,j in ((0,1),(0,2),(1,2))]))
                    auc=lambda hi,lo: float(np.mean(np.array(hi)[:,None]>np.array(lo)[None,:])+0.5*np.mean(np.array(hi)[:,None]==np.array(lo)[None,:]))
                    def guess(tris):
                        hit=tot=0
                        for t in tris:
                            for si in range(3):
                                cs=SEAT_W[si]; best=max(cs,key=lambda x:sc(tuple(x if j==si else t[j] for j in range(3))))
                                hit+=tuple(best if j==si else t[j] for j in range(3)) in ok; tot+=1
                        return hit/tot
                    hh=[sc(t) for t in heard[::max(1,len(heard)//200)]]; pl=[sc(t) for t in plaus]; od=[sc(t) for t in odd]; hf=[sc(t) for t in half]
                    rows.append(dict(話=T,語=n,形=form,本数=K,席=seat,回=p,覚え=round(auc(hh,od),3),広がり=round(auc(pl,od),3),半分変との区別=round(auc(pl,hf),3),
                                     聞いたとありそう=round(auc(hh,pl),3),当て=round(guess(plaus[::max(1,len(plaus)//40)]),3)))
    return rows,dict(話=T,語=n,聞いた=len(heard),ありそう=len(plaus),変=len(odd),半分変=len(half))
FORMS=["今の波に足す","今の波に足す・共通の分を引く","今の波に足す・ほかを遠ざける","元の波を足す","元の波を足す・たまたまの分を引く","元の波を足す・夜に数え直す（偏り）"]
if __name__=="__main__":
    out=[];info=[]
    for T in (10,40,160):
        t0=time.time(); r,i=run(T,(1,2,4,8,16,64,256),(1,5),FORMS); out+=r; info.append(i); print(i,round(time.time()-t0),flush=True)
    json.dump(dict(世界=info,rows=out),open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False)
