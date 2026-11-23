"""I49q: 外れた時に周波数を足す形（持ち主の案、未確定）を、聞く前に当てる流れの小さい世界で数える（乱数なし）。
使い方: python tools/i49q_small_world_grow.py   （画面に出すだけ）
手で置く物: 足した周波数の上での席の位相（3 等分）、元の点と足した分の点をそのまま足す事、上限の本数、使い切ったら古い順に重ねる事。
"""
import sys, math, numpy as np
import os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import i49q_small_world_stream as W
TH=np.exp(2j*np.pi*np.arange(3)/3.0)
class Grow(W.Vec):
    """元の向きを足す・席あり の上に、足した周波数（P）を持つ。when: "外れ"=答えが外れた 3 つ組だけ足す、"毎回"=全部足す、"外れが多い語"=外れが当たりより多い語が居る時だけ。
    cmax 本を使い切ったら、古い順に同じ周波数へ重ねる（上書きはしない。足すだけ）。"""
    def __init__(s,n,B,cmax,when):
        super().__init__(n,B,False,True); s.P=np.zeros((n,cmax),dtype=complex); s.cmax=cmax; s.when=when; s.k=0; s.added=0
        s.h=np.zeros(n); s.m=np.zeros(n); s.last=[None]*3
    def ask(s,si,ids,cands):
        sc=super().ask(si,ids,cands)
        aim=sum(s.P[ids[sj]]*np.conj(TH[sj]) for sj in range(3) if sj!=si)
        pv=np.real((s.P[cands]*np.conj(TH[si]))@np.conj(aim))
        pv[np.isin(cands,[ids[sj] for sj in range(3) if sj!=si])]=0
        tot=sc+pv; tot[sc==-1]=-1
        s.last[si]=bool(tot.max()>0 and cands[int(np.argmax(tot))]==ids[si])
        return tot
    def learn(s,ids):
        super().learn(ids)
        for si in range(3):
            if s.last[si]: s.h[ids[si]]+=1
            else: s.m[ids[si]]+=1
        miss=not all(s.last)
        go=dict(毎回=True,外れ=miss).get(s.when)
        if go is None: go=miss and any(s.m[a]>s.h[a] for a in ids)
        if go:
            c=s.k%s.cmax; s.k+=1; s.added+=1
            for si,a in enumerate(ids): s.P[a,c]+=TH[si]
if __name__=="__main__":
    talks=W.make(); words=sorted({w for _,ws in talks for w in ws}); IDX={w:i for i,w in enumerate(words)}
    B=W.base_dense(words,256)
    for name,mk in W.FORMS[:2]:
        print(name, W.stream(mk(len(words),B),talks,IDX)[0])
    for when in ("毎回","外れ","外れが多い語"):
        for cmax in (16,64,256,1024,4096,16384):
            g=Grow(len(words),B,cmax,when); a,per,parts=W.stream(g,talks,IDX)
            print("足す:%-8s 上限 %5d 本  当て %.3f  足した回数 %5d  席 %s  100 本ごと %s"%(when,cmax,a,g.added,[float(x) for x in per],[float(x) for x in parts]))
