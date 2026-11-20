"""語板に「高い周波数の帯」を紐付けて、語と語の間を周波数のうねりで覚える小さな試し（持ち主の案、未確定）。

1. 語板から 100 語ほどを抜き出し、自作の正しい文（内容語の並び）を作る。どの語も 2 文以上に出る。
2. 語ごとに、属性の帯（品詞・B・C・D・E・G）ごとに 100 個の周波数の値（複素数）を紐付ける（語板とは別の表）。
   帯の中の 100 個の周波数は、語の間の 1 歩を 1 として、搬送 F0 = 1,000 ＋ ずれ ν_j（j = 0..99、ν_j = j × W / 100）。
   語の値 = その語の属性ごとの「属性の形」（100 個の値）を属性の重みで足した物。属性の形は 2 通り:
     乱数    属性ごとに乱数の位相（今の語板と同じ考え。属性が少し違うと全く別の形になる）
     山 σ    属性を、語板全体での使われ方の近さで 1 列に並べ（近い順につなぐ）、その順で帯の中の周波数に置き、
             幅 σ の山（ガウス）にする。近い属性は近い周波数に来て、山が重なる（なだらかさを周波数で作る）
3. 覚え方（うねりを合わせる）: 行の中の語 i（位置 p）と語 k（位置 q）の組ごとに、語 i の覚えの表 R_i に
   「語 k の値 × e^(−i 2π f (q − p))」を足す（距離 q − p を周波数ごとの位相のうねりで持つ）。
   q < p の組（マイナス位置）も足すので、2 語目からも 1 語目を覚える（「マイナス位置なし」= q > p だけ、も並べる）。
4. 読み方（共振）: 位置 p に候補 c を置いた時の共振 = Σ_{ほかの語 i} Re⟨R_i, c の値 × e^(−i 2π f (p − p_i))⟩。
   検波: 帯ごとに |⟨·⟩|（包絡）を取って足す読み方も並べる（搬送の位相が合わなくても読める）。
   1,000 倍の搬送は、語の位置が整数の時は e^(−i 2π 1,000 d) = 1 なので、ずれ ν だけの物と全く同じになる（確かめる）。
測る物:
  A 学習した文の語の戻り（位置ごとに候補 100 語から 1 位）、マイナス位置あり／なし
  B 学習に無い、大体正しい文と、おかしな文の、本当の語の共振がノイズより上か（ノイズ = 語を行の間でばらして覚えた表 20 通り）
  C 読む位置のずれ（0.25・0.5・1・2 歩、と搬送の 1 周期より小さいずれ 0.00025・0.0005 歩）への強さ。W と属性の形ごと
  D なだらかさ: 2 語の属性の近さと値の近さの合い方（順位の相関）、属性を「いちばん近いほかの属性」に 1 つ取り替えた時の値の近さ
  E 行を 1 本の波にして戻す（位置ごとの位相で重ねる）。行全体が τ ずれた時の戻りと、ずれを周波数で探す
    （帯の下 1/4 の周波数だけ・包絡で粗く → 全部の周波数・実部で細かく）
手で置く物: 帯ごとの周波数の数 100、F0 = 1,000、W の並び、σ の並び、ずれの並び、ノイズの回数 20、探す刻み、帯の属性の数（語板の上位 4）。
python tools/i48_freq_board.py --content .../content_boards.json --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib

import numpy as np

# 自作の文と、その内容語（原形）の並び。
LINES = [
    ("友達と公園で会う", "友達 公園 会う"), ("母が台所で料理を作る", "母 台所 料理 作る"),
    ("父は毎朝新聞を読む", "父 毎朝 新聞 読む"), ("子供が公園で遊ぶ", "子供 公園 遊ぶ"),
    ("猫が窓のそばで寝る", "猫 窓 寝る"), ("犬が庭を走る", "犬 庭 走る"),
    ("学生が図書館で本を読む", "学生 図書館 本 読む"), ("先生が学生に英語を教える", "先生 学生 英語 教える"),
    ("雨が降るので傘を持つ", "雨 降る 傘 持つ"), ("駅で電車を待つ", "駅 電車 待つ"),
    ("友達に手紙を書く", "友達 手紙 書く"), ("母が店でパンを買う", "母 店 パン 買う"),
    ("父は車で会社に行く", "父 車 会社 行く"), ("子供が川で魚を見る", "子供 川 魚 見る"),
    ("鳥が空を飛ぶ", "鳥 空 飛ぶ"), ("春に花が咲く", "春 花 咲く"),
    ("夏は海で泳ぐ", "夏 海 泳ぐ"), ("冬は山に雪が降る", "冬 山 雪 降る"),
    ("朝にコーヒーを飲む", "朝 コーヒー 飲む"), ("夜に音楽を聞く", "夜 音楽 聞く"),
    ("学生が自転車で学校に行く", "学生 自転車 学校 行く"), ("医者が病院で働く", "医者 病院 働く"),
    ("先生が黒板に字を書く", "先生 黒板 字 書く"), ("猫が魚を食べる", "猫 魚 食べる"),
    ("犬と公園を散歩する", "犬 公園 散歩"), ("友達と映画を見る", "友達 映画 見る"),
    ("海で写真を撮る", "海 写真 撮る"), ("母が庭に花を植える", "母 庭 花 植える"),
    ("子供が犬と遊ぶ", "子供 犬 遊ぶ"), ("父がコーヒーを飲む", "父 コーヒー 飲む"),
    ("静かな部屋で本を読む", "静か 部屋 本 読む"), ("店で新しい服を買う", "店 新しい 服 買う"),
    ("古い家に住む", "古い 家 住む"), ("電車が駅に着く", "電車 駅 着く"),
    ("バスで学校に行く", "バス 学校 行く"), ("友達に手紙を送る", "友達 手紙 送る"),
    ("おいしい料理を食べる", "おいしい 料理 食べる"), ("熱いお茶を飲む", "熱い お茶 飲む"),
    ("高い山は美しい", "高い 山 美しい"), ("川の水が冷たい", "川 水 冷たい"),
    ("青い空を鳥が飛ぶ", "青い 空 鳥 飛ぶ"), ("部屋の窓を開ける", "部屋 窓 開ける"),
    ("医者が病気を治す", "医者 病気 治す"), ("毎日英語を習う", "毎日 英語 習う"),
    ("夜に星を見る", "夜 星 見る"), ("雪は白い", "雪 白い"),
    ("学校で先生と話す", "学校 先生 話す"), ("病院で薬をもらう", "病院 薬 もらう"),
    ("週末に家族と旅行に行く", "週末 家族 旅行 行く"), ("冬は寒い", "冬 寒い"),
    ("夏は暑い", "夏 暑い"), ("友人と歌を歌う", "友人 歌 歌う"),
    ("机の上に紙を置く", "机 紙"), ("牛乳と卵を買う", "牛乳 卵 買う"),
    ("皿を洗う", "皿 洗う"), ("疲れたので休む", "疲れる 休む"),
    ("友人が家に来る", "友人 家"), ("家族で旅行の写真を見る", "家族 旅行 写真 見る"),
    ("台所で皿を洗う", "台所 皿 洗う"), ("机で手紙を書く", "机 手紙 書く"),
    ("新しい机を買う", "新しい 机 買う"), ("春は暑い日もある", "春 暑い"),
    ("毎朝牛乳を飲む", "毎朝 牛乳 飲む"), ("卵の料理を作る", "卵 料理 作る"),
    ("黒板の字を紙に書く", "黒板 字 紙 書く"), ("星が美しい", "星 美しい"),
    ("お茶が冷たい", "お茶 冷たい"), ("古い歌を聞く", "古い 歌 聞く"),
    ("散歩して疲れる", "散歩 疲れる"), ("休みに病院に行く", "休む 病院 行く"),
    ("薬で病気を治す", "薬 病気 治す"), ("自転車で会社に行く", "自転車 会社 行く"),
    ("バスを待つ", "バス 待つ"), ("寒い朝に雪が降る", "寒い 朝 雪 降る"),
    ("傘を店で買う", "傘 店 買う"), ("おいしいパンを食べる", "おいしい パン 食べる"),
    ("熱いコーヒーを飲む", "熱い コーヒー 飲む"), ("高い服を買う", "高い 服 買う"),
    ("音楽を習う", "音楽 習う"), ("映画の写真を撮る", "映画 写真 撮る"),
    ("毎日植えた花を見る", "毎日 植える 花 見る"), ("雨の日は図書館で本を読む", "雨 図書館 本 読む"),
    ("先生に話す", "先生 話す"), ("電車で友人に会う", "電車 友人 会う"),
    ("家族が家に住む", "家族 家 住む"), ("駅に着く", "駅 着く"),
    ("ドアを開ける", "開ける"), ("鳥が窓に来る", "鳥 窓"),
    ("週末は海に行く", "週末 海 行く"), ("寝る前に音楽を聞く", "寝る 音楽 聞く"),
    ("犬が走る", "犬 走る"), ("子供が泳ぐ", "子供 泳ぐ"),
    ("水を飲む", "水 飲む"), ("青い花が咲く", "青い 花 咲く"),
    ("白い猫が寝る", "白い 猫 寝る"), ("父が働く", "父 働く"),
    ("母に手紙を送る", "母 手紙 送る"), ("学生が英語を教える", "学生 英語 教える"),
    ("店でもらう", "店 もらう"), ("静かな夜", "静か 夜"),
    ("重い荷物を持つ", "持つ"), ("駅で新聞を買う", "駅 新聞 買う"), ("子供が歌を歌う", "子供 歌 歌う"),
    ("車を洗う", "車 洗う"), ("傘を持つ", "傘 持つ"),
]
# 学習に無い、大体正しい文（語板では似た語があるが、この組み合わせは学習に無い）
UNSEEN_OK = [
    ("犬が川で泳ぐ", "犬 川 泳ぐ"), ("猫が庭を走る", "猫 庭 走る"), ("学生が映画を見る", "学生 映画 見る"),
    ("父がパンを食べる", "父 パン 食べる"), ("子供が花を見る", "子供 花 見る"), ("先生が本を読む", "先生 本 読む"),
    ("友達が海で泳ぐ", "友達 海 泳ぐ"), ("鳥が山を飛ぶ", "鳥 山 飛ぶ"), ("母が牛乳を飲む", "母 牛乳 飲む"),
    ("医者が英語を話す", "医者 英語 話す"), ("友人に写真を送る", "友人 写真 送る"), ("夜に雨が降る", "夜 雨 降る"),
    ("学生がバスを待つ", "学生 バス 待つ"), ("先生が手紙を書く", "先生 手紙 書く"), ("子供が卵を食べる", "子供 卵 食べる"),
]
# おかしな文（語は全部学習にあるが、組み合わせが変）
UNSEEN_BAD = [
    ("魚が新聞を読む", "魚 新聞 読む"), ("パンが空を飛ぶ", "パン 空 飛ぶ"), ("雪が料理を作る", "雪 料理 作る"),
    ("傘が映画を泳ぐ", "傘 映画 泳ぐ"), ("机が音楽を歌う", "机 音楽 歌う"), ("星が皿を洗う", "星 皿 洗う"),
    ("薬が駅で遊ぶ", "薬 駅 遊ぶ"), ("紙が病気を食べる", "紙 病気 食べる"),
]
BANDS = ("品詞", "B", "C", "D", "E", "G")
NC = 100
F0 = 1000.0
TOPK = 4
WS = (0.25, 0.5, 1.0)
LAYOUTS = (("乱数", None), ("山 σ1", 1.0), ("山 σ3", 3.0))
SHIFTS = (0.0, 0.00025, 0.0005, 0.25, 0.5, 1.0, 2.0)
N_NOISE = 20
UPOS = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ"}


def words_attrs(cb, lemma):
    w = cb[lemma]
    out = {"品詞": {UPOS.get(w["pos"], "NOUN"): 1.0}}
    for b in BANDS[1:]:
        A = w["attrs"].get(b, {})
        top = sorted(A.items(), key=lambda z: -z[1])[:TOPK]
        if top:
            out[b] = dict(top)
    return out


def chain_order(names, vecs):
    """使われ方の近い属性を順につなぐ（いちばん多く使われる物から、まだ使っていない中でいちばん近い物へ）。"""
    V = vecs / np.maximum(1e-9, np.linalg.norm(vecs, axis=1, keepdims=True))
    C = V @ V.T
    left = set(range(len(names)))
    cur = int(np.argmax(np.linalg.norm(vecs, axis=1)))
    order = [cur]
    left.discard(cur)
    while left:
        L = np.array(sorted(left))
        cur = int(L[np.argmax(C[cur, L])])
        order.append(cur)
        left.discard(cur)
    return [names[i] for i in order], C


class Board:
    def __init__(self, cb_all, lemmas, layout, sigma, W, carrier=False):
        self.lemmas = lemmas
        self.li = {w: i for i, w in enumerate(lemmas)}
        self.att = {w: words_attrs(cb_all, w) for w in lemmas}
        self.nu = np.tile(np.arange(NC) * W / NC, len(BANDS))                    # 検波した後のずれ
        self.f = self.nu + (F0 if carrier else 0.0)
        self.near_attr = {}
        self.shapes = {}
        for bi, b in enumerate(BANDS):
            names = sorted({a for w in lemmas for a in self.att[w].get(b, {})})
            if not names:
                continue
            if sigma is None:
                for a in names:
                    r = np.random.default_rng(zlib.crc32((b + "|" + a).encode()))
                    self.shapes[(b, a)] = np.exp(2j * np.pi * r.random(NC)) / np.sqrt(NC)
            else:
                vec = usage_vectors(cb_all, b, names)
                order, C = chain_order(names, vec)
                pos = {a: (k + 0.5) * NC / len(order) for k, a in enumerate(order)}
                j = np.arange(NC)
                for a in names:
                    s = np.exp(-0.5 * ((j - pos[a]) / sigma) ** 2)
                    self.shapes[(b, a)] = s / np.linalg.norm(s)
                ii = {a: i for i, a in enumerate(names)}
                for a in names:
                    c = C[ii[a]].copy()
                    c[ii[a]] = -9
                    self.near_attr[(b, a)] = names[int(np.argmax(c))]
        self.S = np.stack([self.value(self.att[w]) for w in lemmas])

    def value(self, att):
        v = np.zeros(len(BANDS) * NC, dtype=np.complex128)
        for bi, b in enumerate(BANDS):
            A = att.get(b)
            if not A:
                continue
            x = sum(wt * self.shapes[(b, a)] for a, wt in A.items())
            n = np.linalg.norm(x)
            if n > 0:
                v[bi * NC:(bi + 1) * NC] = x / n
        return v / max(1e-12, np.linalg.norm(v))

    def ph(self, d):
        return np.exp(-2j * np.pi * self.f * d)

    def learn(self, lines, minus=True):
        R = np.zeros_like(self.S)
        for ln in lines:
            ids = [self.li[w] for w in ln]
            for p, i in enumerate(ids):
                for q, k in enumerate(ids):
                    if q == p or (not minus and q < p):
                        continue
                    R[i] += self.S[k] * self.ph(q - p)
        return R

    def scores(self, R, ids, p, delta=0.0, env=False):
        """位置 p（ほかの語は ids の位置）に置く候補ごとの共振。delta = 読む位置のずれ。"""
        Q = np.zeros(self.S.shape[1], dtype=np.complex128)
        for pi, i in enumerate(ids):
            if pi != p:
                Q += np.conj(R[i]) * self.ph(p - pi + delta)
        Z = self.S * Q[None, :]
        if env:
            return np.abs(Z.reshape(len(self.S), len(BANDS), NC).sum(2)).sum(1)
        return np.real(Z.sum(1))


def usage_vectors(cb_all, b, names, top=10000):
    words = sorted(cb_all.values(), key=lambda w: -w.get("freq", 0))[:top]
    ni = {a: i for i, a in enumerate(names)}
    M = np.zeros((len(names), len(words)), dtype=np.float32)
    for k, w in enumerate(words):
        for a, v in w["attrs"].get(b, {}).items():
            if a in ni:
                M[ni[a], k] = v
    return M


def recall(bd, R, lines, delta=0.0, env=False):
    h = h5 = n = 0
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        if len(ids) < 2:
            continue
        for p, t in enumerate(ids):
            sc = bd.scores(R, ids, p, delta, env)
            r = int((sc > sc[t]).sum())
            h += r == 0
            h5 += r < 5
            n += 1
    return round(h / n, 4), round(h5 / n, 4)


def resonance_vs_noise(bd, R, Rn, lines):
    """本当の語の共振を、ノイズの表（語を行の間でばらして覚えた物）での共振の並びと比べる。"""
    z, above, rank0 = [], [], []
    for ln in lines:
        ids = [bd.li[w] for w in ln]
        for p, t in enumerate(ids):
            s = bd.scores(R, ids, p)[t]
            nz = np.array([bd.scores(Rk, ids, p)[t] for Rk in Rn])
            z.append((s - nz.mean()) / max(1e-12, nz.std()))
            above.append(s > np.quantile(nz, 0.95))
            sc = bd.scores(R, ids, p)
            rank0.append(int((sc > sc[t]).sum()))
    return dict(本当の語の共振_ノイズからの離れ_平均z=round(float(np.mean(z)), 2),
                ノイズの上位5パーセントより上=round(float(np.mean(above)), 4),
                候補100語での1位=round(float(np.mean(np.array(rank0) == 0)), 4),
                候補100語での上位5=round(float(np.mean(np.array(rank0) < 5)), 4))


def line_wave(bd, ids, tau):
    return sum(bd.S[i] * bd.ph(p + tau) for p, i in enumerate(ids))


def read_wave(bd, X, n, tau_hat=0.0, env=True):
    out = []
    for p in range(n):
        Y = X * np.conj(bd.ph(p + tau_hat))
        Z = np.conj(bd.S) * Y[None, :]
        sc = np.abs(Z.reshape(len(bd.S), len(BANDS), NC).sum(2)).sum(1) if env else np.real(Z.sum(1))
        out.append(sc)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cb_all = {}
    for w in json.load(open(a.content, encoding="utf-8")):
        if w["name"] not in cb_all or w.get("freq", 0) > cb_all[w["name"]].get("freq", 0):
            cb_all[w["name"]] = w
    train = [s.split() for _, s in LINES]
    ok = [s.split() for _, s in UNSEEN_OK]
    bad = [s.split() for _, s in UNSEEN_BAD]
    lemmas = sorted({w for s in train for w in s})
    use = {w: sum(w in s for s in train) for w in lemmas}
    seen_pairs = {(s[p], s[q]) for s in train for p in range(len(s)) for q in range(len(s)) if p != q}
    res = dict(語=len(lemmas), 学習の文=len(train), 語ごとの出る文の数_最小=min(use.values()),
               語ごとの出る文の数_平均=round(float(np.mean(list(use.values()))), 2),
               一度だけの語=[w for w, c in use.items() if c < 2],
               学習に無い組の割合_大体正しい文=round(float(np.mean([(s[p], s[q]) not in seen_pairs for s in ok for p in range(len(s)) for q in range(len(s)) if p != q])), 4),
               学習に無い組の割合_おかしな文=round(float(np.mean([(s[p], s[q]) not in seen_pairs for s in bad for p in range(len(s)) for q in range(len(s)) if p != q])), 4),
               型={})
    for s in ok + bad:
        for w in s:
            assert w in use, w
    rng = np.random.default_rng(7)
    flat = [w for s in train for w in s]
    noise_lines = []
    for _ in range(N_NOISE):
        perm = list(rng.permutation(flat))
        nl, k = [], 0
        for s in train:
            nl.append(perm[k:k + len(s)])
            k += len(s)
        noise_lines.append(nl)

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for lay, sigma in LAYOUTS:
        for W in WS:
            t1 = time.time()
            bd = Board(cb_all, lemmas, lay, sigma, W)
            R = bd.learn(train)
            R1 = bd.learn(train, minus=False)
            Rn = [bd.learn(nl) for nl in noise_lines]
            r = {}
            r["A_学習した文_1位・上位5"] = recall(bd, R, train)
            r["A_マイナス位置なし_1位・上位5"] = recall(bd, R1, train)
            r["B_学習した文"] = resonance_vs_noise(bd, R, Rn, train)
            r["B_学習に無い大体正しい文"] = resonance_vs_noise(bd, R, Rn, ok)
            r["B_おかしな文"] = resonance_vs_noise(bd, R, Rn, bad)
            r["C_ずれ_そのまま_1位"] = {str(d): recall(bd, R, train, d)[0] for d in SHIFTS}
            r["C_ずれ_検波(帯ごとの包絡)_1位"] = {str(d): recall(bd, R, train, d, env=True)[0] for d in SHIFTS}
            bdc = Board(cb_all, lemmas, lay, sigma, W, carrier=True)
            Rc = bdc.learn(train)
            r["C_1000倍の搬送のまま_そのまま_1位"] = {str(d): recall(bdc, Rc, train, d)[0] for d in SHIFTS}
            r["C_1000倍の搬送のまま_検波_1位"] = {str(d): recall(bdc, Rc, train, d, env=True)[0] for d in SHIFTS}
            r["C_搬送あり・なしで共振が同じ(整数の位置)"] = bool(np.allclose(bd.scores(R, [bd.li[w] for w in train[1]], 0),
                                                         bdc.scores(Rc, [bd.li[w] for w in train[1]], 0), atol=1e-6))
            # D なだらかさ
            if W == WS[0]:
                Aw = attr_matrix(bd)
                ca = Aw @ Aw.T
                cv = np.real(bd.S @ np.conj(bd.S).T)
                iu = np.triu_indices(len(lemmas), 1)
                r["D_属性の近さと値の近さの順位の相関"] = round(spearman(ca[iu], cv[iu]), 4)
                bins = [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]
                r["D_属性の近さの帯ごとの値の近さ_平均"] = {f"{lo}〜{hi:.1f}": round(float(cv[iu][(ca[iu] >= lo) & (ca[iu] < hi)].mean()), 4)
                                                     if ((ca[iu] >= lo) & (ca[iu] < hi)).any() else None for lo, hi in bins}
                if bd.near_attr:
                    keep = []
                    for w in lemmas:
                        for b in BANDS[1:]:
                            A = bd.att[w].get(b)
                            if not A:
                                continue
                            top = max(A, key=A.get)
                            nb = bd.near_attr[(b, top)]
                            if nb in A:
                                continue
                            att2 = {k: dict(v) for k, v in bd.att[w].items()}
                            att2[b][nb] = att2[b].pop(top)
                            keep.append(float(np.real(np.vdot(bd.S[bd.li[w]], bd.value(att2)))))
                    r["D_いちばん強い属性を近い属性に取り替えた時の値の近さ_平均"] = round(float(np.mean(keep)), 4)
                    br = Board(cb_all, lemmas, "乱数", None, W)
                    keep2 = []
                    for w in lemmas:
                        for b in BANDS[1:]:
                            A = br.att[w].get(b)
                            if not A:
                                continue
                            top = max(A, key=A.get)
                            nb = bd.near_attr[(b, top)]
                            if nb in A:
                                continue
                            att2 = {k: dict(v) for k, v in br.att[w].items()}
                            att2[b][nb] = att2[b].pop(top)
                            if (b, nb) not in br.shapes:
                                rr = np.random.default_rng(zlib.crc32((b + "|" + nb).encode()))
                                br.shapes[(b, nb)] = np.exp(2j * np.pi * rr.random(NC)) / np.sqrt(NC)
                            keep2.append(float(np.real(np.vdot(br.S[br.li[w]], br.value(att2)))))
                    r["D_同じ取り替えを乱数の形でした時_平均"] = round(float(np.mean(keep2)), 4)
            # E 行を 1 本の波にして戻す
            e = {}
            lt = [s for s in train if len(s) >= 2]
            for tau in (0.0, 0.25, 0.5, 1.0):
                h = n = 0
                for s in lt:
                    ids = [bd.li[w] for w in s]
                    g = read_wave(bd, line_wave(bd, ids, tau), len(ids)).argmax(1)
                    h += int((g == np.array(ids)).sum())
                    n += len(ids)
                e[f"ずれ{tau}_探さない"] = round(h / n, 4)
            er = np.random.default_rng(11)
            taus = er.uniform(-1.5, 1.5, len(lt))
            errs, h, n = [], 0, 0
            narrow = np.zeros(len(BANDS) * NC, dtype=bool)
            narrow[np.tile(np.arange(NC) < NC // 4, len(BANDS))] = True
            for s, tau in zip(lt, taus):
                ids = [bd.li[w] for w in s]
                X = line_wave(bd, ids, tau)
                # 粗く: 帯の下 1/4 の周波数だけ（うねりが緩く、ずれに強い）で −2〜+2 を 0.1 刻み
                grid = np.arange(-2, 2.0001, 0.1)
                Xn = np.where(narrow, X, 0)
                g1 = [read_wave(bd, Xn, len(ids), t).max(1).sum() for t in grid]
                t1_ = grid[int(np.argmax(g1))]
                # 細かく: 全部の周波数で ±0.1 を 0.01 刻み。位相まで合わせる読み方（包絡ではなく実部）で山の頂を探す
                fine = t1_ + np.arange(-0.1, 0.1001, 0.01)
                g2 = [read_wave(bd, X, len(ids), t, env=False).max(1).sum() for t in fine]
                th = fine[int(np.argmax(g2))]
                errs.append(abs(th - tau))
                g = read_wave(bd, X, len(ids), th).argmax(1)
                h += int((g == np.array(ids)).sum())
                n += len(ids)
            e["ずれを探して読む_語の戻り"] = round(h / n, 4)
            e["ずれを探して読む_ずれの読み違い_平均(歩)"] = round(float(np.mean(errs)), 4)
            e["ずれを探して読む_0.05歩より外した行の割合"] = round(float(np.mean(np.array(errs) > 0.05)), 4)
            r["E_行を1本の波にして戻す(検波)"] = e
            r["秒"] = round(time.time() - t1, 1)
            nm = f"{lay}・W {W}"
            res["型"][nm] = r
            print(nm, json.dumps(r, ensure_ascii=False), flush=True)
            save()
    res["秒"] = round(time.time() - t0)
    save()


def attr_matrix(bd):
    names = sorted({(b, a) for w in bd.lemmas for b, A in bd.att[w].items() for a in A})
    ni = {x: i for i, x in enumerate(names)}
    M = np.zeros((len(bd.lemmas), len(names)))
    for k, w in enumerate(bd.lemmas):
        for b, A in bd.att[w].items():
            nv = np.sqrt(sum(v * v for v in A.values()))
            for a_, v in A.items():
                M[k, ni[(b, a_)]] = v / nv
    return M / np.maximum(1e-12, np.linalg.norm(M, axis=1, keepdims=True))


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
