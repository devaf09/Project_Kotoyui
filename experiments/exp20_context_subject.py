"""実験20（非公開の材料）: 前の発言（流れ）と人物の手がかりを足して、主語を隠した発言の人称を当てる（実験19 の続き、持ち主の計画 (2) の入口）

実験19 と同じ問・同じ分け方に、前の発言を足す:
  流れの手がかり（① の続き）: 前の発言に人の名前（〜さん・〜様・〜くん・〜ちゃん・〜先生、片仮名の名前）や第三者の語（彼・彼女・あいつ…）がある → 第三者、
  前の発言が問い → 話し手（問われた人が自分のことを答える）、前の発言に「あなた・君」→ 話し手、前の発言に「私・僕・俺」→ 相手
  袋: 主語を消した文の袋 + 前の発言の袋（別の印を付けて同じ 4096 個に散らす）
以下は実験19 の説明:
「話し手・相手・第三者」のどれかを当てる — 誰の話かを特定する力の物差し（持ち主の計画 (1)、2026-11-04）

着想: 持ち主「文章上で誰かを暗黙的に（格が明示されていない）場合、誰の話をしているかあまり正確じゃない」。日本語は主語を省く。
      主語がはっきり書かれた発言（「私は」「あなたが」「彼は」…）の主語を隠し、残りの手がかりで人称を当てさせれば、答えの付いた問が大量に作れる。
      ①（話者間の関係・言葉の手がかり）と ②（文法解析）を、同じ物差しで比べられる。
問: 発言のうち、人称の主語（下の表の語 + は・が・も）がちょうど 1 つあるもの。その主語（語と助詞）を消した文 → 1 人称（話し手）/ 2 人称（相手）/ 3 人称（第三者）。
    類の数をそろえる（教え・試しとも、少ない類に合わせて切る）。
方法:
  rules : ① 手がかりの決まり（言語学で知られた省略された主語の手がかり）。点を足して一番大きい人称:
          敬語（いらっしゃる・おっしゃる・なさる… → 相手か第三者）、謙譲（参る・申す・いたす・伺う → 話し手）、
          〜てくれる・〜てもらう（→ 主語は相手・第三者、または話し手）、〜てあげる（→ 話し手）、〜たい（→ 話し手）、〜たがる・〜そう・〜らしい（→ 第三者）、
          気持ちの形容詞の言い切り（嬉しい・悲しい → 話し手）、問い（〜か？・〜の？ → 相手）、頼み・命令（〜て下さい・〜なさい → 相手）、
          〜と言っていた・〜みたい（→ 第三者）
  nb    : 数えた表（語尾・文字の 2 つ組の袋、実験01 の袋）
  light : 光の 2DWIG（同じ袋を 64 × 64 の光の場に、Net2D の板 16 枚。実験01・15 と同じ）
  いつも同じ答え: 3 類をそろえたので 33.3%
材料: build_persona_pairs.py の発言（RealPersonaChat + 持ち主が集めた Web 小説の会話文）。教え・試しの分け方はそのまま。
結果: ../results/exp19.txt、例 ../docs/exp19_examples.json（非公開）
実行（Project_Kotoyui の直下で）: python experiments/exp19_hidden_subject.py --device cuda
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

import torch

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
sys.path.insert(0, os.path.join(KOTO, "experiments"))
from exp01_wig_bag import L, bag, p2, to_field  # noqa: E402

PAIRS = os.path.join(PRIV, "data", "persona", "pairs.jsonl")
PERSONS = ["話し手", "相手", "第三者"]
SUBJ = {0: r"私|わたし|僕|ぼく|俺|おれ|オレ|あたし|わし|自分|我々|私たち|僕たち|俺たち",
        1: r"あなた|貴方|君|きみ|キミ|お前|おまえ|あんた|貴様|そちら|あなたたち|君たち|お前ら",
        2: r"彼|彼女|あいつ|あの人|あの子|あの方|奴|やつ|彼ら|彼女たち|あいつら"}
SUBJ_RE = {k: re.compile(r"(?:^|(?<=[、。！？…\s「]))(" + v + r")(は|が|も)") for k, v in SUBJ.items()}

# ① 手がかりの決まり: (正規表現, 人称ごとの点)
RULES = [
    (r"(いらっしゃ|おっしゃ|なさ[っるいれら]|召し上が|ご覧にな|お越し|くださ[っるいれ]|お[^\s、。]{1,4}にな[るっり])", (0, 1, 1)),
    (r"(参り|参る|申し|申す|いたし|致し|伺い|伺う|存じ|拝見|おり[まし])", (2, 0, 0)),
    (r"てくれ|てくださ|てもら[っうえい]", (0, 1, 1)),
    (r"てあげ|てやっ|てやる", (2, 0, 0.5)),
    (r"たい(です|な|んだ|の|けど|から|と|。|！|$)", (2, 0, 0)),
    (r"たが[っるり]|そうだ|そうです|らしい|みたい|ようだ|と言って|って言って|んだって|そうな", (0, 0, 2)),
    (r"(嬉しい|悲しい|寂しい|怖い|楽しい|痛い|辛い|眠い|欲しい)(です|な|よ|。|！|$)", (1.5, 0, 0)),
    (r"(か[？?。]?|の[？?]|かな[？?]?|[？?])$", (0, 2, 0)),
    (r"(て下さい|てください|なさい|てくれ[。！]?$|ろ[。！]?$|ませんか|ないか[？?]?$)", (0, 2, 0)),
    (r"(よね|でしょ)[。？?]?$", (0, 1, 0)),
    (r"(た|ました)[。！]?$", (0.5, 0, 0.5)),
]
RULES = [(re.compile(p), w) for p, w in RULES]


NAME = re.compile(r"([一-龥ァ-ヶー]{1,6})(さん|様|さま|くん|君|ちゃん|先生|殿)|[ァ-ヶー]{2,8}(?=[はがもをにと、])")
THIRD = re.compile(SUBJ[2])


def context_scores(prev):
    s = [0.0, 0.0, 0.0]
    if prev is None:
        return s
    if NAME.search(prev) or THIRD.search(prev):
        s[2] += 1.5
    if re.search(r"(か[？?。]?|の[？?]|[？?])$", prev):
        s[0] += 1.5
    if re.search(SUBJ[1], prev):
        s[0] += 1.0
    if re.search(SUBJ[0], prev):
        s[1] += 1.0
    return s


def rule_scores(t):
    s = [0.0, 0.0, 0.0]
    for p, w in RULES:
        if p.search(t):
            for k in range(3):
                s[k] += w[k]
    return s


def mask(t):
    """人称の主語がちょうど 1 つの発言 → (主語を消した文, 人称)。当てはまらなければ None。"""
    hits = [(k, m) for k, r in SUBJ_RE.items() for m in r.finditer(t)]
    if len(hits) != 1:
        return None
    k, m = hits[0]
    rest = (t[:m.start()] + t[m.end():]).strip("、 ")
    return (rest, k) if len(rest) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_class", type=int, default=60000)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    seen = set()
    tr, te = {k: [] for k in range(3)}, {k: [] for k in range(3)}
    for line in open(PAIRS, encoding="utf-8"):
        r = json.loads(line)
        for t in (r["prev"], r["reply"]):
            if t in seen:
                continue
            seen.add(t)
            mt = mask(t)
            if mt:
                (tr if r["split"] == "train" else te)[mt[1]].append((mt[0], mt[1], t, r["src"], r["prev"] if t == r["reply"] else None))
    print("問の数（教え・試し）", {PERSONS[k]: (len(tr[k]), len(te[k])) for k in range(3)}, f"{time.perf_counter() - t0:.0f} 秒", flush=True)
    n_tr = min(a.max_per_class, min(len(v) for v in tr.values()))
    n_te = min(5000, min(len(v) for v in te.values()))
    trs, tes = [], []
    for k in range(3):
        rng.shuffle(tr[k])
        rng.shuffle(te[k])
        trs += tr[k][:n_tr]
        tes += te[k][:n_te]
    rng.shuffle(trs)
    yt = torch.tensor([x[1] for x in tes])
    runs = []

    def report(name, pred, extra=None):
        pred = torch.as_tensor(pred)
        per = {PERSONS[k]: 100.0 * (pred[yt == k] == k).float().mean().item() for k in range(3)}
        r = dict(method=name, acc=100.0 * (pred == yt).float().mean().item(), per_class=per, **(extra or {}))
        runs.append(r)
        print(name, round(r["acc"], 1), {k: round(v, 1) for k, v in per.items()}, flush=True)
    # ① 手がかりの決まり（同点は 話し手 > 相手 > 第三者 の順。手がかりが無い問の割合も出す）
    rs = [rule_scores(x[0]) for x in tes]
    report("① 手がかりの決まり（実験19）", [max(range(3), key=lambda k: (s[k], -k)) for s in rs], dict(no_cue=sum(max(s) == 0 for s in rs) / len(rs)))
    rs = [[a_ + b_ for a_, b_ in zip(rule_scores(x[0]), context_scores(x[4]))] for x in tes]
    report("① + 流れの手がかり", [max(range(3), key=lambda k: (s[k], -k)) for s in rs], dict(no_cue=sum(max(s) == 0 for s in rs) / len(rs), with_prev=sum(x[4] is not None for x in tes) / len(tes)))
    # 数えた表（袋）
    X = bag([x[0] for x in trs])
    y = torch.tensor([x[1] for x in trs])
    Xt = bag([x[0] for x in tes])
    cnt = torch.zeros(3, L * L)
    for k in range(3):
        cnt[k] = X[y == k].float().sum(0)
    logp = torch.log((cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True))
    report("数えた表（その発言だけ、実験19）", (Xt.float() @ logp.T).argmax(1))
    pv = lambda xs: bag([("〓" + x[4]) if x[4] else "" for x in xs])         # 前の発言の袋（印を付けて同じ 4096 個に散らす）
    X = X + pv(trs)
    Xt = Xt + pv(tes)
    cnt = torch.zeros(3, L * L)
    for k in range(3):
        cnt[k] = X[y == k].float().sum(0)
    logp = torch.log((cnt + 0.5) / (cnt + 0.5).sum(1, keepdim=True))
    report("数えた表（その発言 + 前の発言）", (Xt.float() @ logp.T).argmax(1))
    # 光の 2DWIG
    dev = torch.device(a.device)
    net = p2.Net2D(L, 3, ["xy"] * 16, patches=16, seed=a.seed).to(dev)
    t1 = time.perf_counter()
    p2.train(net, X, y, steps=a.steps, lr=0.01, batch_size=256, seed=a.seed, to_input=to_field, schedule="cosine")
    net.eval()
    with torch.no_grad():
        S = torch.cat([net(to_field(Xt[i:i + 2000]).to(dev)).cpu() for i in range(0, len(Xt), 2000)])
    report("光の 2DWIG（その発言 + 前の発言）", S.argmax(1), dict(train_s=time.perf_counter() - t1))
    # 光 + ①（決まりの点を足す。重みは試しの前半で選び、後半で測る）
    R = torch.tensor(rs)
    LS = torch.log_softmax(S, 1)
    half = len(yt) // 2
    best = max((0.0, 0.25, 0.5, 1.0, 2.0), key=lambda w: ((LS[:half] + w * R[:half]).argmax(1) == yt[:half]).float().mean().item())
    report("光の 2DWIG + ① + 流れ", (LS + best * R).argmax(1), dict(weight=best))
    NBt = torch.log_softmax(Xt.float() @ logp.T, 1)
    best2 = max((0.0, 0.25, 0.5, 1.0, 2.0), key=lambda w: ((NBt[:half] + w * R[:half]).argmax(1) == yt[:half]).float().mean().item())
    report("数えた表 + ① + 流れ", (NBt + best2 * R).argmax(1), dict(weight=best2))
    rec = dict(exp="20", **vars(a), n_train=len(trs), n_test=len(tes), runs=runs, total_s=time.perf_counter() - t0)
    with open(os.path.join(PRIV, "results", "exp20.txt"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pred = (LS + best * R).argmax(1)
    ex = [dict(masked=x[0], orig=x[2], gold=PERSONS[x[1]], pred=PERSONS[int(pred[i])], rules=rs[i]) for i, x in enumerate(tes[:60])]
    with open(os.path.join(PRIV, "docs", "exp20_examples.json"), "w", encoding="utf-8") as f:
        json.dump(ex, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
