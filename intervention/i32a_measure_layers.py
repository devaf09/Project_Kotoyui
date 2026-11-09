"""介入の研究 実験I32a: つなぎ・述部の形は板で当てられるか（ほかの作品で数え、自分の作品は引く）

つなぎ: 行の中で述語 p が述語 q に係る所の、p のあとの機能語の列の末尾。いつも て・いちばん多い・p・q・(p,q) の当たりを比べる。
述部の形: 根の述語の形の印（た・丁寧・否定・相）が、前の行・同じ作品のこれまでの行に引っ張られるか。
Web 小説は _wn。結果は非公開の置き場だけ。本文はコードに書かない。
実行: python intervention/i32a_measure_layers.py <解析済み JSON>
"""
import json,collections,sys,math
PRED={'VERB','ADJ'}
FUNDEP={'aux','mark','case','cop','fixed','cc'}
CONNPOS={'SCONJ','ADP','PART','CCONJ'}
VOICE={'する','れる','られる','せる','させる'}
TAILAUX={'たら','だら','なら','たり','だり','ずに','ず','に','ば'}
def chain(T,kids,p):
  fun=sorted([k for k in kids[p['i']] if k['i']>p['i'] and k['dep'] in FUNDEP and k['pos']!='PUNCT'],key=lambda k:k['i'])
  # 連続しているものだけ
  out=[];nxt=p['i']+1
  for k in fun:
    if k['i']!=nxt: break
    out.append(k);nxt+=1
  return out
def layers(T,kids,p):
  ch=chain(T,kids,p)
  voice=[];i=0
  while i<len(ch) and ch[i]['lemma'] in VOICE and ch[i]['pos']=='AUX': voice.append(ch[i]['lemma']);i+=1
  rest=ch[i:]
  j=len(rest)
  while j>0 and rest[j-1]['pos'] in CONNPOS: j-=1
  conn=''.join(k['text'] for k in rest[j:])
  form=rest[:j]
  if not conn:
    if form and form[-1]['text'] in TAILAUX: conn=form[-1]['text'];form=form[:-1]
  return dict(voice='+'.join(voice),form='+'.join(k['lemma'] for k in form),conn=conn,surf=p['text'],lemma=p['lemma'],
              end=(ch[-1]['i'] if ch else p['i']))
def line_preds(ln):
  T=ln['toks'];kids=collections.defaultdict(list)
  for t in T:
    if t['head']!=t['i']: kids[t['head']].append(t)
  res=[]
  for t in T:
    if t['pos'] in PRED or (t['dep']=='ROOT'):
      L=layers(T,kids,t);L.update(i=t['i'],dep=t['dep'],head=t['head'],pos=t['pos'])
      res.append(L)
  return res

def conn_and_form(d):
  CLS={'て':'順','連用':'順','てから':'順','に':'目的','ため':'理由','から':'理由','ので':'理由','し':'並べ',
   'が':'逆接','けど':'逆接','けれど':'逆接','のに':'逆接','ても':'逆接','でも':'逆接','ながらも':'逆接',
   'ながら':'同時','つつ':'同時','と':'条件','ば':'条件','たら':'条件','なら':'条件','ず':'否定','たり':'並べ'}
  def cls(c): return CLS.get(c,'他')
  def isq(t): return t.lstrip('　 ').startswith(('「','『','（'))
  def tense(f):
    return dict(た='た' in f.split('+'),丁寧=any(x in f for x in('ます','です')),否定=any(x in f.split('+') for x in('ない','ぬ')),相='てる' in f)
  # 材料
  A=[];B=[]
  for wi,w in enumerate(d):
    prevf=None;prevq=None
    for ln in w['lines']:
      P=line_preds(ln);byi={L['i']:L for L in P}
      for L in P:
        if L['dep']=='advcl' and L['head'] in byi and byi[L['head']]['pos'] in('VERB','ADJ'):
          A.append((wi,L['lemma'],byi[L['head']]['lemma'],L['conn'] or '連用'))
      roots=[L for L in P if L['dep']=='ROOT' and L['pos'] in('VERB','ADJ')]
      q=isq(ln['text'])
      if roots:
        f=roots[0]['form'];tf=tense(f)
        if prevf is not None: B.append((wi,roots[0]['lemma'],q,prevq,prevf,tf))
        prevf=tf;prevq=q
  def wb(pairs_by_ctx):
    pass
  # --- つなぎ ---
  def counts(items,key):
    tot=collections.defaultdict(collections.Counter);per=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for it in items:
      k=key(it);tot[k][it[3]]+=1;per[it[0]][k][it[3]]+=1
    return tot,per
  def loo(tot,per,wi,k):
    c=tot.get(k,collections.Counter()).copy();c.subtract(per[wi].get(k,{}))
    return +c
  keys={'g':lambda it:0,'p':lambda it:it[1],'h':lambda it:it[2],'ph':lambda it:(it[1],it[2])}
  C={n:counts(A,f) for n,f in keys.items()}
  labels=sorted({it[3] for it in A})
  def WB(c,lower):
    n=sum(c.values());t=len(c)
    if n==0: return lower
    lam=n/(n+t)
    return {x:lam*c.get(x,0)/n+(1-lam)*lower[x] for x in labels}
  res=collections.defaultdict(lambda:[0,0,0])
  for it in A:
    wi=it[0];g=loo(*C['g'],wi,0);n=sum(g.values())
    pg={x:(g.get(x,0)+0.5)/(n+0.5*len(labels)) for x in labels}
    pp=WB(loo(*C['p'],wi,it[1]),pg);ph_=WB(loo(*C['h'],wi,it[2]),pg)
    mix={x:math.sqrt(pp[x]*ph_[x]) for x in labels}
    s=sum(mix.values());mix={x:v/s for x,v in mix.items()}
    P={'いつも て':{'て':1},'いちばん多い':pg,'前の述語':pp,'次の述語':ph_,
       '前→組':WB(loo(*C['ph'],wi,(it[1],it[2])),pp),'両方→組':WB(loo(*C['ph'],wi,(it[1],it[2])),mix)}
    for name,dist in P.items():
      best=max(dist,key=dist.get);top3=sorted(dist,key=dist.get,reverse=True)[:3]
      r=res[name];r[0]+=best==it[3];r[1]+=cls(best)==cls(it[3]);r[2]+=it[3] in top3
  N=len(A)
  print('つなぎ 件数',N,'（前の述語→次の述語、行の中）')
  for k,(a,b,c) in res.items(): print(f'{k}\t当たり {a/N:.3f}\t型の当たり {b/N:.3f}\t上位3 {c/N:.3f}')
  cc=collections.Counter(cls(it[3]) for it in A);print('型の割合',{k:round(v/N,3) for k,v in cc.most_common()})
  # --- 述部の形 ---
  print('\n述部の形 件数',len(B),'（前の行の根の形 → この行の根の形）')
  for feat in ['た','丁寧','否定','相']:
    tot=collections.defaultdict(collections.Counter);per=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    ks={'g':lambda b:0,'台詞':lambda b:b[2],'前の行':lambda b:(b[2],b[3],b[4][feat]),'語':lambda b:b[1]}
    T={n:collections.defaultdict(collections.Counter) for n in ks};PW={n:collections.defaultdict(lambda:collections.defaultdict(collections.Counter)) for n in ks}
    for b in B:
      for n,f in ks.items(): T[n][f(b)][b[5][feat]]+=1;PW[n][b[0]][f(b)][b[5][feat]]+=1
    def get(n,b):
      c=T[n].get(ks[n](b),collections.Counter()).copy();c.subtract(PW[n][b[0]].get(ks[n](b),{}));return +c
    acc=collections.Counter()
    for b in B:
      y=b[5][feat]
      g=get('g',b);pg={v:(g.get(v,0)+.5)/(sum(g.values())+1) for v in(True,False)}
      labels_=(True,False)
      def wb2(c,lo):
        n=sum(c.values());t=len(c)
        if n==0:return lo
        l=n/(n+t);return {v:l*c.get(v,0)/n+(1-l)*lo[v] for v in labels_}
      pq=wb2(get('台詞',b),pg);pprev=wb2(get('前の行',b),pq);pw=wb2(get('語',b),pq)
      pboth={v:pprev[v]*pw[v]/pq[v] for v in labels_}
      for name,dist in {'いちばん多い':pg,'台詞か地の文か':pq,'前の行に合わせる':pprev,'語だけ':pw,'前の行と語':pboth}.items():
        acc[name]+=max(dist,key=dist.get)==y
    base=sum(b[5][feat] for b in B)/len(B)
    print(f'[{feat}] ある割合 {base:.3f} | '+' | '.join(f'{k} {v/len(B):.3f}' for k,v in acc.items()))


def form_by_history(d):
  def isq(t): return t.lstrip('　 ').startswith(('「','『','（'))
  F={'た':lambda f:'た' in f,'丁寧':lambda f:'ます' in f or 'です' in f,'否定':lambda f:'ない' in f or 'ぬ' in f,'相':lambda f:'てる' in f}
  rows=[]
  for wi,w in enumerate(d):
    hist={q:collections.Counter() for q in(True,False)};hn={True:0,False:0};prev=None
    for ln in w['lines']:
      P=line_preds(ln);roots=[L for L in P if L['dep']=='ROOT' and L['pos'] in('VERB','ADJ')]
      if not roots: continue
      fs=roots[0]['form'].split('+');q=isq(ln['text'])
      y={k:f(fs) for k,f in F.items()}
      if prev is not None and hn[q]>=1:
        rows.append((wi,q,prev,{k:hist[q][k]/hn[q] for k in F},hn[q],y))
      for k in F: hist[q][k]+=y[k]
      hn[q]+=1;prev=(q,y)
  print('件数',len(rows))
  for k in F:
    ys=[r[5][k] for r in rows];base=sum(ys)/len(ys)
    same=[r for r in rows if r[2][0]==r[1]]
    p1=sum(r[5][k] for r in same if r[2][1][k])/max(1,sum(1 for r in same if r[2][1][k]))
    p0=sum(r[5][k] for r in same if not r[2][1][k])/max(1,sum(1 for r in same if not r[2][1][k]))
    # 予測: 作品のこれまでの割合（同じ台詞/地の文）を確からしさにする。対数損失で比べる
    def ll(p,y): p=min(max(p,1e-3),1-1e-3); return -math.log(p if y else 1-p)
    L0=sum(ll(base,r[5][k]) for r in rows)/len(rows)
    Lh=sum(ll((r[3][k]*r[4]+base)/(r[4]+1),r[5][k]) for r in rows)/len(rows)  # これまでの行（ほかの作品の割合を 1 行ぶんの重みで混ぜる）
    # 釣り合いの当たり: 作品の割合 > ほかの割合 なら ある
    tp=sum(1 for r in rows if r[5][k] and (r[3][k]*r[4]+base)/(r[4]+1)>=0.5);fn=sum(1 for r in rows if r[5][k])
    tn=sum(1 for r in rows if not r[5][k] and (r[3][k]*r[4]+base)/(r[4]+1)<0.5);fp=len(rows)-fn
    print(f'[{k}] ある割合 {base:.3f} | 前の行が「ある」→ある {p1:.3f} / 前の行が「ない」→ある {p0:.3f} | 損失 いつも同じ {L0:.3f} → これまでの行 {Lh:.3f} | ある行の当たり {tp/fn:.3f}・ない行の当たり {tn/fp:.3f}')


if __name__=='__main__':
  d=json.load(open(sys.argv[1],encoding='utf-8'))
  conn_and_form(d)
  form_by_history(d)
