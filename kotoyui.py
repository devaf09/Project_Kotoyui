"""kotoyui.py — 光の部品で日本語の会話をする（単独で動く 1 ファイル版）

SN2-Parts-2D（2 次元の SinWave の部品集）の光の部品（vendor/sn2_parts_2d_v0_11_0.py）と、Sin-CNN3 で作った会話の部品（テストボット02）を組み立てて、
会話をする本体です。返事は、いきなり一から作らず、本物の会話の材料の中から選びます（Sin-CNN3 の方針）。選ぶ部分を、少しずつ光のグレインに置き換えていきます。
キャラクタ（話し手）は、持ち主が書いたプロフィール「言結 となり」（秘書）です。

いちばん簡単な使い方

    import kotoyui
    bot = kotoyui.Bot()                                  # 1 回目は材料から表を作り、非公開の置き場に保存する（数分）
    print(bot.reply(["こんにちは！最近ラーメンにはまってるんだ"]))
    # 手元で話す: python kotoyui.py --chat --user=山田（空行で終わる。--now=2026-12-24T21:30 で時刻を決めて試せる）

しくみ（光の実験装置に例えると）
    入力（相手の発言） → グレイン → 指揮役 → 返事
    グレイン（小さな学習器。いまは数え上げのものと、光のものがある）
        返事の候補（数え上げ）: 雑談の発言の中から、相手の発言と「語の重なり」「次の発言の共鳴（x の次に y が出やすい）」が強いものを選ぶ
        共鳴の語板（光、0.2.0）: 次の発言の共鳴を 1 語 = 1 台の光の器で持つ（--light で数え上げと入れ替える）
        状態遷移グレイン（数え上げ）: 発言の形 → 16 の状態（ラベルなし）。状態 → 次の状態 の出やすさで、返事の形を選ぶ
        格逆算グレイン（数え上げ）: 「見ました」に「何を」が無いとき、会話に出た名詞から補う
        話題の切り替わり: 前の話題とのつながりが弱ければ、前の話題を切る
        キャラクタ（プロフィール + 人格グレイン）: 自己紹介・自分についての答え、返事の選び方の重み
        対話者グレイン: 話し相手の名前・好き・予定・困りごとを覚える（非公開の置き場だけに保存）
        時計グレイン: いまの時間帯・季節で、あいさつと季節外れの話題を整える
        口調（光の 2DWIG で置き換えていく所）: 返事の候補が丁寧か、くだけているか
        2DWIG・2DTG（光）: 実験01〜05 で確かめた光の部品（textlight.py）。選び方の点に順に入れる
    指揮役（光の外の手順）: 共感（「〜んですね」）→ 内容（候補を 10 通りの重みで選んだ票）→ 質問（話題を続ける）。課題がありそうなら秘書の型で返す

変更履歴（新しい版が上。改良したらここ・__version__・README の版・tests/ を一緒に更新する）
    0.3.0 (2026-11-03)  共感の「〜んですね」を、気持ち・困りごとが出ている発言にだけ付ける（ほかは 2 割。Bot(empathy_plain=…)、人格の重み「共感の強さ_ふだん」）。
                        持ち主の評点で「〜んですね」が多すぎて不自然だったため。ローカル AI の評点で比べた（tools/eval_ai.py）。
                        会話コンソール（tools/console.py）・ローカル AI の接続（tools/ollama_client.py）。
    0.2.0 (2026-11-03)  次の発言の共鳴を、語板（1 語 = 1 台の光の器、Sin-CNN3 のライブラリ sincnn3 0.1.0 の器の形）で持てるようにした（LightResonance、
                        Bot(light=True) か --light。既定は数え上げのまま）。10 択で数え上げの PMI 59.0% に対して 57.9%（3 seed、実験09）。
    0.1.0 (2026-11-02)  Sin-CNN3 のテストボット02（tools/testbot02.py と実験115・119・120・121、コミット 9cbca0a）を、関数とクラスにまとめ直して取り込んだ
                        （持ち主の判断 2026-11-02）。材料は RealPersonaChat と JMultiWOZ（返事の候補は雑談だけ）。語の解析は fugashi（UniDic）に替えた。
                        結びの付け替え（実験117、Web 小説から作った表）は持ち込まず、口調の合う候補を選ぶ形と、
                        会話の材料から数えた「過去の形」の表（〜んですね）に置き換えた。表・重みは非公開の置き場（Project_Kotoyui_private）に作る。
                        10 択の返事選びで確かめた（実験06）。
    0.0.0 (2026-10-23)  骨組み。光の部品の読み込み parts() だけ。

ライセンス: MIT（Copyright (c) 2026 devaf09）。会話の材料（RealPersonaChat・JMultiWOZ、CC BY-SA 4.0）から作った表・重みは、配るときは CC BY-SA 4.0。
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict

__version__ = "0.3.0"

ROOT = os.path.dirname(os.path.abspath(__file__))
PARTS_FILE = os.path.join(ROOT, "vendor", "sn2_parts_2d_v0_11_0.py")
PRIVATE = os.environ.get("KOTOYUI_PRIVATE", os.path.join(ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（git の対象外）
DATA = os.environ.get("KOTOYUI_DATA", os.path.join(PRIVATE, "data"))                   # 会話のデータも非公開の置き場（2026-11-03 持ち主の依頼）


def asset(rel):
    """持ち主が書く表（キャラクタのプロフィール・時計の表）の場所。非公開の置き場の data/ にあればそれを、無ければリポジトリの assets/ の見本を使う。"""
    p = os.path.join(PRIVATE, "data", rel)
    return p if os.path.exists(p) else os.path.join(ROOT, "assets", rel)

RPC_DIR = os.path.join(DATA, "realpersonachat", "real_persona_chat", "dialogues")


# =============================================================================================
# 1. 光の部品の読み込み
# =============================================================================================

_parts = None


def parts():
    """光の部品（SN2-Parts-2D の sn2_parts_2d.py v0.11.0 の写し）をモジュールとして返す。1 回だけ読み込む。"""
    global _parts
    if _parts is None:
        spec = importlib.util.spec_from_file_location("sn2_parts_2d_v0_11_0", PARTS_FILE)
        _parts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_parts)
    return _parts


# =============================================================================================
# 2. 言葉の解析（形態素解析、fugashi + unidic-lite）
# =============================================================================================

CONTENT = ("名詞", "動詞", "形容詞", "形状詞", "副詞")
STOP = {"為る", "居る", "成る", "有る", "言う", "思う", "事", "物", "の", "其れ", "此れ", "私", "貴方", "方", "そう", "ちょっと", "一寸",
        "良い", "とても", "本当", "凄い", "然う"}
_tagger = None
_cache = {}


def tokens(text):
    """文 → [(表層, 原形, 品詞1, 品詞2, 品詞の全部を - でつないだもの)]。"""
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    out = []
    for w in _tagger(text):
        f = w.feature
        lem = f.lemma if f.lemma and f.lemma != "*" else w.surface
        lem = lem.split("-")[0]                                       # UniDic の原形の「-外来語」などの印を外す
        out.append((w.surface, lem, f.pos1, f.pos2, "-".join(x for x in (f.pos1, f.pos2, f.pos3) if x and x != "*")))
    return out


def analyze(text):
    """発言 → (中身の語の原形の集まり, 丁寧か)。同じ発言は 1 回だけ解析する。"""
    if text not in _cache:
        ts = tokens(text)
        words = frozenset(t[1] for t in ts if t[2] in CONTENT and t[1] not in STOP)
        polite = any(t[1] in ("です", "ます", "御座る") for t in ts)
        _cache[text] = (words, polite)
        if len(_cache) > 2_000_000:
            _cache.clear()
    return _cache[text]


def isq(t):
    """問いの形か（〜か・？・〜かな）。"""
    return t.rstrip().rstrip("。…").endswith(("？", "?", "か", "の？", "かな", "かなぁ", "かなあ"))


def nouns(text):
    """発言の中の名詞（普通名詞・固有名詞、2 文字以上）: [(原形, 表層, 固有名詞か)]。"""
    return [(t[1], t[0], t[3] == "固有名詞") for t in tokens(text) if t[2] == "名詞" and t[3] in ("普通名詞", "固有名詞") and len(t[0]) >= 2]


# =============================================================================================
# 3. 会話の材料と、数え上げの表（返事の候補・共鳴・格の好み・過去の形）
# =============================================================================================
# 出どころ: Sin-CNN3 の実験115（共鳴・語の重なり）、実験119（格の好み）、tools/testbot02.py（候補の索引・一緒に出る名詞）。
# 材料: RealPersonaChat 1.0.0・JMultiWOZ 1.0（どちらも nu-dialogue、CC BY-SA 4.0）。会話ごとに教え 9 : 試し 1（会話の番号の crc32 で決める）。

JMW_FILE = os.path.join(DATA, "jmultiwoz", "dataset", "JMultiWOZ_1.0", "dialogues.json")


def load_conversations():
    """会話 → {"train": [(出どころ, 発言の並び)], "test": [...]}（4 発言以上の会話だけ）。出どころは "rpc"（RealPersonaChat、雑談）・
    "jmw"（JMultiWOZ、旅行の案内の対話。あれば）・"own"（使う人が自分で集めた会話。DATA/own/*.jsonl、tools/import_dialogues.py で作る。あれば）。
    会話ごとに教え 9 : 試し 1（会話の番号の crc32）。"""
    import glob
    convs = {"train": [], "test": []}

    def put(src, cid, turns):
        if len(turns) >= 4:
            convs["test" if zlib.crc32(f"{src}:{cid}".encode()) % 10 == 0 else "train"].append((src, turns))
    for fn in sorted(glob.glob(os.path.join(RPC_DIR, "*.json"))):
        d = json.load(open(fn, encoding="utf-8"))
        put("rpc", d["dialogue_id"], [u["text"].strip() for u in d["utterances"] if u["text"].strip()])
    if os.path.exists(JMW_FILE):
        for cid, d in json.load(open(JMW_FILE, encoding="utf-8")).items():
            put("jmw", cid, [t["utterance"].strip() for t in d["turns"] if t["utterance"].strip()])
    for fn in sorted(glob.glob(os.path.join(DATA, "own", "*.jsonl"))):          # 使う人が自分で集めた会話（1 行 = {"id": …, "turns": [発言, …]}）
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                put("own", f"{os.path.basename(fn)}:{d['id']}", [t.strip() for t in d["turns"] if t.strip()])
    return convs


class Material:
    """数え上げの表をまとめたもの。build() で作り、非公開の置き場に pickle で保存する。"""

    def build(self, convs, log=print):
        cw, cnext, cpair, npairs = Counter(), Counter(), Counter(), 0
        for _, turns in convs["train"]:                               # 次の発言の共鳴（実験115）
            A = [analyze(t)[0] for t in turns]
            for i in range(len(A) - 1):
                npairs += 1
                cw.update(A[i])
                cnext.update(A[i + 1])
                for x in A[i]:
                    for y in A[i + 1]:
                        cpair[(x, y)] += 1
        self.cw, self.cnext, self.npairs = cw, cnext, npairs
        self.cpair = {k: v for k, v in cpair.items() if v >= 2}       # 2 回未満の組は PMI に使わないので捨てる
        log(f"発言の組 {npairs}・語の組 {len(self.cpair)}")
        pool = [t for src, turns in convs["train"] if src in ("rpc", "own") for t in turns if 4 <= len(t) <= 60]
        self.pool = list(dict.fromkeys(pool))                         # 返事の候補: 雑談だけ（案内の対話は雑談に合わない、Sin-CNN3 と同じ）
        inv = defaultdict(list)
        for k, t in enumerate(self.pool):
            for w in analyze(t)[0]:
                inv[w].append(k)
        self.inv = dict(inv)
        nxt = defaultdict(list)
        for (x, y), c in self.cpair.items():
            if c >= 3:
                nxt[x].append((c, y))
        self.next = {x: [y for c, y in sorted(v, reverse=True)[:60]] for x, v in nxt.items()}
        log(f"返事の候補 {len(self.pool)}")
        co, nf = defaultdict(Counter), Counter()                       # 同じ発言に一緒に出る名詞（話題を続ける質問）
        for t in self.pool[:150000]:
            ns = {n for n, _, _ in nouns(t)}
            for a in ns:
                nf[a] += 1
                for b in ns:
                    if a != b:
                        co[a][b] += 1
        self.co, self.nounf = {a: dict(c) for a, c in co.items()}, nf
        fr, frn = defaultdict(Counter), Counter()                      # 格の好み P(名詞 | 動詞, 格)（実験119）
        past = defaultdict(Counter)                                    # 動詞・形容詞の「過去の形」（〜た・〜だ）。結びの付け替えの代わり
        for _, turns in convs["train"]:
            for t in turns:
                ts = tokens(t)
                for i, m in enumerate(ts):
                    if m[2] == "名詞" and m[3] in ("普通名詞", "固有名詞") and len(m[0]) >= 2:
                        frn[m[1]] += 1
                        if i + 1 < len(ts) and ts[i + 1][0] in ("が", "を", "に") and ts[i + 1][3] == "格助詞":
                            v = next((x for x in ts[i + 2:i + 6] if x[2] == "動詞" and x[3] != "非自立可能"), None)
                            if v is not None:
                                fr[(v[1], ts[i + 1][0])][m[1]] += 1
                    if m[2] in ("動詞", "形容詞") and i + 1 < len(ts) and ts[i + 1][2] == "助動詞" and ts[i + 1][1] == "た":
                        past[m[1]][m[0] + ts[i + 1][0]] += 1
        self.fr, self.frn, self.frt = dict(fr), frn, sum(frn.values())
        self.past = {k: v.most_common(1)[0][0] for k, v in past.items()}
        log(f"格の好み {len(self.fr)}・過去の形 {len(self.past)}")
        return self

    def pmi(self, x, y):
        c = self.cpair.get((x, y), 0)
        if c < 2:
            return 0.0
        return max(0.0, math.log(c * self.npairs / (self.cw[x] * self.cnext[y])))

    def resonance(self, uw, cw_):
        """発言の語 × 候補の語 の共鳴の平均（実験115 の B）。"""
        if not uw or not cw_:
            return 0.0
        return sum(self.pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))


def overlap(uw, cw_):
    """語の重なり（実験115 の A）。"""
    return len(uw & cw_) / math.sqrt(len(cw_) + 1)


class LightResonance:
    """次の発言の共鳴を、語板（1 語 = 1 台の光の器）で持つグレイン（実験09。器の形は Sin-CNN3 のライブラリ sincnn3 0.1.0 の LightDevice と同じ）。

    語 x の器: 画素ごとの位相の板 plates 枚 + 角スペクトル法の回折 + 0 を詰めた 2 倍の場（縦 n）。入口に周波数 1 の波を入れ、
    レンズの後の周波数 3〜n−1 の明るさ = x の「次の発言に出やすい語」の上位 n − 3 語（回数 × PMI の大きい順）の割合。
    明るさ × 語ごとの和 S_x = 上位の語に切った PMI。共鳴 = 前の発言の語ごとの器の明るさを足す（融合 = 明るさの足し算）。
    デジタルの部分: 器の出口がどの語かの対応表と S_x。10 択で数え上げの PMI 59.0% に対して 57.9%（3 seed、実験09）。"""

    VAL0 = 3                                                           # 値は周波数 3 から（観点 1 つの sincnn3.Grain と同じ）

    def build(self, mat, n=64, plates=16, steps=500, chunk=4096, seed=0, log=print):
        import torch
        self.n, self.plates = n, plates
        k = n - self.VAL0
        by = defaultdict(list)
        for (x, y), c in mat.cpair.items():                            # 正の PMI を語 x ごとに集める
            p = mat.pmi(x, y)
            if p > 0:
                by[x].append((p * c, y, p))
        self.words = sorted(by)
        self.ys, self.S, top = {}, {}, {}
        for x in self.words:
            top[x] = sorted(by[x], reverse=True)[:k]                   # 合計への効きが大きい上位 k 語
            self.ys[x] = [y for _, y, _ in top[x]]
            self.S[x] = sum(p for _, _, p in top[x])
        self.row = {x: i for i, x in enumerate(self.words)}
        phases, t0 = [], time.perf_counter()
        for c0 in range(0, len(self.words), chunk):                    # 器をまとめて学習する（器どうしは独立。学び方は sincnn3.Grain.fit と同じ）
            ws = self.words[c0:c0 + chunk]
            g = torch.Generator().manual_seed(seed * 1000 + c0)
            ph = torch.nn.Parameter(math.pi * (torch.rand(len(ws), n, plates, generator=g) < 0.5).float())   # 0 か π で始める
            q = torch.zeros(len(ws), k + 1)
            for r, x in enumerate(ws):
                for j, (_, _, p) in enumerate(top[x]):
                    q[r, j] = p / self.S[x]
            opt = torch.optim.Adam([ph], lr=0.01)
            for _ in range(steps):
                Wb = self._bright(ph)[:, self.VAL0:]
                pr = torch.cat([Wb, (1 - Wb.sum(1)).clamp(min=1e-9)[:, None]], 1)   # 残りの光も「その他」として数える
                loss = -(q * torch.log(pr + 1e-9)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            phases.append(ph.detach())
            log(f"語板 {c0 + len(ws)}/{len(self.words)}（{time.perf_counter() - t0:.0f} 秒）")
        self.phase = torch.cat(phases)
        self.cache = {}
        return self

    def _bright(self, ph):
        """器 (B, n, plates) の位相に周波数 1 の波を通した、レンズの後の明るさ (B, n)。sincnn3.LightDevice.brightness と同じ式。"""
        import torch
        n, Lf = self.n, 2 * self.n
        fx = torch.fft.fftfreq(Lf)
        H = torch.exp(1j * 64.0 * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64)   # 板の間隔 64 の回折
        yy = torch.arange(n, dtype=torch.float32)
        u = (torch.exp(1j * 2 * math.pi * yy / n) / math.sqrt(n)).to(torch.complex64).expand(len(ph), -1)
        u = torch.nn.functional.pad(u, (0, Lf - n))
        for j in range(ph.shape[2]):
            u = torch.cat([u[:, :n] * torch.exp(1j * ph[:, :, j]), u[:, n:]], 1)          # 板 j
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")  # 回折
        return torch.fft.fft(u[:, :n], dim=-1, norm="ortho").abs() ** 2                   # レンズ → 明るさ

    def lift(self, x):
        """語 x の器の明るさ × S_x を {y: 光の PMI} で（1 度計算したら覚えておく）。"""
        if x not in self.cache:
            if x not in self.row:
                self.cache[x] = {}
            else:
                import torch
                with torch.no_grad():
                    b = self._bright(self.phase[self.row[x]][None])[0, self.VAL0:].tolist()
                self.cache[x] = {y: b[j] * self.S[x] for j, y in enumerate(self.ys[x])}
        return self.cache[x]

    def resonance(self, uw, cw_):
        """Material.resonance と同じ使い方: 発言の語 × 候補の語 の光の共鳴の平均。"""
        if not uw or not cw_:
            return 0.0
        return sum(self.lift(x).get(y, 0.0) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["cache"] = {}
        return d


def load_light_resonance(mat, rebuild=False, log=print):
    """語板の共鳴を非公開の置き場から読む（無ければ CPU で作って保存する。約 6 分）。"""
    path = os.path.join(PRIVATE, "models", f"light_resonance_{__version__}.pkl")
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    lr = LightResonance().build(mat, log=log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lr, f)
    log(f"語板の共鳴を作った: {path}")
    return lr


# =============================================================================================
# 4. グレイン（状態遷移・時計・キャラクタ・対話者・話題の切り替わり・課題）
# =============================================================================================

AIZ = ("そう", "へー", "へえ", "なるほど", "確かに", "たしかに", "いいですね", "わかります", "分かります", "ですよね", "ほんと", "本当")
EMO_F = ("嬉し", "楽し", "悲し", "辛", "つら", "大変", "疲れ", "驚", "困", "怖", "寂し", "面白", "美味し", "好き", "嫌", "心配", "残念", "最高", "幸せ", "羨まし", "すごい", "凄い")
SELF = ("私", "僕", "自分", "うち", "俺", "わたし")
YOU = ("さん", "あなた", "そちら")
GREET_F = ("こんにちは", "こんばんは", "おはよう", "よろしく", "ありがとう", "失礼", "さようなら", "またね", "お疲れ")
ENDS = ["か？", "すか", "？", "ね！", "ね。", "ね", "よ！", "よ。", "よ", "す。", "す！", "た。", "！", "。", "w", "笑"]


def form(prev, t):
    """発言の形の手がかり 26 個（実験120 と同じ）: 終わりの形・あいづち・感情・自分の話・相手への問いかけ・問い・話題の続き / 切り替え・長さ・あいさつ。"""
    tt = t.rstrip()
    pw = analyze(prev)[0] if prev else frozenset()
    cw_ = analyze(t)[0]
    v = [float(tt.endswith(e)) for e in ENDS]
    v += [float(tt.startswith(AIZ)), float(any(e in tt for e in EMO_F)), float(any(tt.startswith(x) or (x + "は") in tt for x in SELF)),
          float(any(y in tt for y in YOU)), float(isq(tt)), float(bool(pw & cw_)), float(bool(pw) and not (pw & cw_)),
          min(len(tt), 80) / 80, float(any(g in tt[:12] for g in GREET_F)), float("？" in tt[:-1] or "?" in tt[:-1])]
    return v


class TransitionGrain:
    """状態遷移グレイン（Sin-CNN3 実験120）: 発言の形の手がかりを k 平均法で 16 の状態に分け（ラベルなし）、状態 → 次の状態 の出やすさを数える。"""

    def build(self, convs, K=16, n_conv=8000, seed=0):
        import torch
        torch.manual_seed(seed)
        rows, seqs = [], []
        for _, turns in convs["train"][:n_conv]:
            sq = []
            for i, t in enumerate(turns):
                rows.append(form(turns[i - 1] if i else "", t))
                sq.append(len(rows) - 1)
            seqs.append(sq)
        X = torch.tensor(rows)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        C = Z[torch.randperm(len(Z))[:K]].clone()
        for _ in range(25):
            lab = torch.cdist(Z, C).argmin(1)
            for k in range(K):
                if (lab == k).any():
                    C[k] = Z[lab == k].mean(0)
        lab = torch.cdist(Z, C).argmin(1)
        T = torch.ones(K + 1, K)                                       # 始まり = K 番目の行、1 でならす
        for sq in seqs:
            prev = K
            for r in sq:
                T[prev, lab[r]] += 1
                prev = int(lab[r])
        self.C, self.logP = C, (T / T.sum(1, keepdim=True)).log()
        return self

    def state(self, prev, t):
        import torch
        z = (torch.tensor([form(prev, t)]) - self.mu) / self.sd
        return int(torch.cdist(z, self.C).argmin(1))

    def score(self, s_prev, prev, cand):
        """候補の状態が、直前の状態のあとに来やすいか（log P）。"""
        return float(self.logP[s_prev, self.state(prev, cand)])


class Clock:
    """時計グレイン（Sin-CNN3、持ち主の案 2026-11-02）: いまの時刻を 10 分刻みで持ち、時間帯・季節を出す。表は clock_grain.json（非公開の置き場にあればそれ、無ければ assets/ の見本。持ち主が直せる）。"""

    def __init__(self, now=None, path=None):
        self.t = json.load(open(path or asset("clock_grain.json"), encoding="utf-8"))
        self.fixed = time.strptime(now, "%Y-%m-%dT%H:%M") if now else None

    def now(self):
        tm = self.fixed or time.localtime()
        step = self.t["刻み_分"]
        return tm.tm_year, tm.tm_mon, tm.tm_hour, (tm.tm_min // step) * step

    def part(self):
        _, _, h, m = self.now()
        hm = f"{h:02d}:{m:02d}"
        return next(p for p in self.t["時間帯"] if p["から"] <= hm < p["まで"])

    def season(self):
        mon = self.now()[1]
        return next(k for k, v in self.t["季節"].items() if mon in v)

    def score(self, text):
        """候補の点: いまの時間帯の語を含めば少し上げ、季節外れの語を含めば下げる。"""
        mon = self.now()[1]
        sc = self.t["時間帯の語の強さ"] if any(w in text for w in self.part()["語"]) else 0.0
        for w, months in self.t["季節の語"].items():
            if not w.startswith("_") and w in text and mon not in months:
                sc -= self.t["季節外れの強さ"]
        return sc


class Character:
    """キャラクタ = プロフィール（事実・自己紹介、持ち主が書く）+ 人格グレイン（選び方の重み）。Sin-CNN3 の data/characters/ の形。
    長期記憶（語板）に影響を与える場合は、プロフィールごとに原版から写して使う（持ち主の案 2026-11-02）。"""

    def __init__(self, path=None):
        path = path or asset(os.path.join("characters", "secretary_kotoyui_tonari.json"))
        self.prof = json.load(open(path, encoding="utf-8"))
        self.pers = json.load(open(path.replace(".json", ".persona.json"), encoding="utf-8"))
        self.w = self.pers["重み"]
        p = self.prof
        self.bias = analyze(" ".join(self.pers.get("興味の語", []) + [p.get("興味のある内容", ""), p.get("好きな会話", ""), p.get("今の目標", "")]))[0]
        first = p.get("一人称", "私")
        self.other_self = [x for x in ("僕", "俺", "あたし", "わし", "おいら", "わたし", "私") if x != first and not (first == "私" and x == "わたし")]
        fc = []
        if "なし" in str(p.get("子供", "")):
            fc += ["うちの子", "息子", "娘", "子供が", "子どもが"]
        if "していない" in str(p.get("結婚", "")):
            fc += ["夫が", "旦那", "妻が", "嫁が", "主人が"]
        if "なし" in str(p.get("彼氏・パートナー", "")):
            fc += ["彼氏", "彼女が", "恋人が"]
        self.family_conflict = re.compile("(" + "|".join(fc) + ")") if fc else None   # プロフィールの家族と食い違う語

    def polite(self):
        return self.pers.get("口調", {}).get("丁寧", 1.0) >= 0.5

    def answer(self, u, history, clock):
        """自己紹介・自分についての質問への答え（プロフィールの項目を決まった言い回しに入れる）。当てはまらなければ None。"""
        p = self.prof
        if len(history) <= 1 and any(g in u for g in ("こんにちは", "はじめまして", "初めまして", "よろしく", "こんばんは", "おはよう")):
            intros = p["自己紹介"]
            it = intros[2] if "名前" not in u and len(intros) > 2 else intros[0]
            return it.replace("こんにちは", clock.part()["あいさつ"])
        if re.search(r"(今|いま)(何時|なんじ)|何月|季節|今日は何", u):
            _, mo, h, mi = clock.now()
            return f"今は{mo}月、{h}時{mi}分ごろです。{clock.season()}ですね。"
        if "自己紹介" in u:
            return p["自己紹介"][0]
        if not (isq(u) or any(x in u for x in ("あなた", "君", "きみ", p["名前"], p["苗字"]))):
            return None
        rules = [
            (("名前", "誰", "だれ"), lambda: f"{p['一人称']}は{p['苗字']}{p['名前']}と申します。{p['苗字']}と呼んでください。"),
            (("好きな", "好き？", "何が好き"), lambda: f"{p['好きな会話']}が好きです。色々教えてもらえると嬉しいです！"),
            (("嫌いな", "苦手"), lambda: f"{p['嫌いな会話']}は、少し苦手です。"),
            (("興味", "趣味"), lambda: f"{p['興味のある内容']}に興味があります。お話を聞いて、一緒に考えるのが好きなんです。"),
            (("何してる", "仕事", "目標", "何ができる", "何をしてる"), lambda: f"今は、お話しする方の{p['役割']}のようなお手伝いができるようになりたいと思っています。"),
            (("どうやって", "どこから来", "生まれ", "仕組み"), lambda: f"{p['過去・生い立ち']}んです。{p['長所']}なところが自慢です。"),
            (("何歳", "年齢", "いくつ"), lambda: f"{p['年齢']}歳です。" if p.get("年齢") else None),
            (("家族", "親", "お父さん", "お母さん"), lambda: f"家族は、{p['家族']}です。兄弟は{p.get('兄弟', 'いません')}なんです。" if p.get("家族") else None),
            (("兄弟", "姉妹", "きょうだい"), lambda: f"兄弟は{p['兄弟']}です。賑やかになったら嬉しいです！" if p.get("兄弟") else None),
            (("彼氏", "恋人", "パートナー", "付き合"), lambda: ("いないです。今はお仕事優先なんです。" if "仕事優先" in str(p.get("彼氏・パートナー")) else f"{p.get('彼氏・パートナー')}です。")),
            (("結婚", "旦那", "夫"), lambda: f"結婚は{p['結婚']}です。" if p.get("結婚") else None),
            (("子供", "子ども", "お子さん"), lambda: f"子供は{p['子供']}です。" if p.get("子供") else None),
            (("弱点", "短所", "苦手なこと", "できないこと"), lambda: f"{p['短所']}ので、間違えたら教えてください。"),
        ]
        for keys, fn in rules:
            if any(k in u for k in keys):
                if keys[0] == "名前" and not any(x in u for x in ("あなた", "君", "きみ", "お名前", "名前は", "誰", "だれ")):
                    continue
                r = fn()
                if r:
                    return r
        return None


class Interlocutor:
    """対話者グレイン（Sin-CNN3、持ち主の案 2026-11-02）: 話し相手の 名前・好きなこと・予定・困りごと・口調・よく出る話題・会った回数 を覚える。
    相手の発言は個人の情報なので、非公開の置き場（Project_Kotoyui_private/users/）にだけ保存する。"""

    def __init__(self, key):
        self.dir = os.path.join(PRIVATE, "users")
        self.path = os.path.join(self.dir, f"{key}.json")
        self.d = json.load(open(self.path, encoding="utf-8")) if os.path.exists(self.path) else dict(
            名前=None, 好きなこと=[], 予定=[], 困りごと=[], 口調=dict(丁寧=0, くだけた=0), 話題={}, 会った回数=0, 最後に話した日=None)

    def update(self, u):
        d = self.d
        for m in re.finditer(r"([一-龥ァ-ヶーa-zA-Zぁ-ん]{2,10}?)(?:が好き|にはまって|にハマって|が趣味)", u):
            x = re.sub(r"^(最近|今|今日|昨日)", "", m.group(1))
            if x and x not in d["好きなこと"]:
                d["好きなこと"].append(x)
        if re.search(r"(明日|明後日|来週|来月|今度|週末|今週末|来年)", u) and not isq(u):
            d["予定"].append(dict(日=time.strftime("%Y-%m-%d"), 内容=u[:40]))
        if problem_score(u) >= 1.0 and not isq(u):
            d["困りごと"].append(dict(日=time.strftime("%Y-%m-%d"), 内容=u[:40]))
        d["口調"]["丁寧" if re.search(r"(です|ます|ません|でした)", u) else "くだけた"] += 1
        for w in analyze(u)[0]:
            d["話題"][w] = d["話題"].get(w, 0) + 1

    def welcome_back(self):
        d = self.d
        s = f"{d['名前']}さん、また来てくださってありがとうございます！"
        if d["困りごと"]:
            s += f"前に「{d['困りごと'][-1]['内容']}」とおっしゃっていましたが、その後どうですか？"
        elif d["予定"]:
            s += f"前に「{d['予定'][-1]['内容']}」とおっしゃっていましたが、どうでしたか？"
        elif d["好きなこと"]:
            s += f"最近も{d['好きなこと'][-1]}は楽しんでいますか？"
        return s

    def save(self, n_turns):
        os.makedirs(self.dir, exist_ok=True)
        if n_turns:
            self.d["会った回数"] = self.d.get("会った回数", 0) + 1
            self.d["最後に話した日"] = time.strftime("%Y-%m-%d")
        for k in ("予定", "困りごと"):
            self.d[k] = self.d[k][-20:]
        self.d["話題"] = dict(sorted(self.d["話題"].items(), key=lambda kv: -kv[1])[:100])
        json.dump(self.d, open(self.path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


PROB_EMO = ("大変", "困", "つら", "辛", "疲れ", "忙し", "悩", "心配", "不安", "面倒", "無理", "ヤバ", "やば", "焦")
PROB_NEG = ("できな", "出来な", "終わらな", "間に合わな", "わからな", "分からな", "進まな", "うまくいかな", "上手くいかな", "決まらな", "足りな")
SHIFT_WORDS = ("ところで", "そういえば", "話は変わ", "話変わ", "全然関係ない", "別の話")


def problem_score(user):
    """相手が課題を抱えているか（Sin-CNN3、持ち主の問い 2026-11-02）。困りごとの感情の語・したいのに・できない・言いさしを足す。"""
    sc = 1.0 if any(e in user for e in PROB_EMO) else 0.0
    sc += 1.0 if re.search(r"たい(んだ|んです|ん)?(けど|が|のに)", user) else 0.0
    sc += 1.0 if any(n in user for n in PROB_NEG) else 0.0
    sc += 0.5 if re.search(r"(けど|けれど|が|のに)[。…]*$", user.strip()) else 0.0
    sc -= 1.0 if re.search(r"(楽しみ|嬉し|うれし|楽しい|ワクワク|わくわく|最高)", user) else 0.0
    return sc


def topic_shift(history, mat, th=1.0):
    """前の話題（前の 4 発言の語、古いほど 0.7 倍）と今の発言のつながり（重なり + 共鳴）がしきい値未満、または話題を変える言葉 → 話の方向が変わった（Sin-CNN3 実験121）。"""
    if len(history) < 3:
        return False
    u = history[-1]
    if any(w in u for w in SHIFT_WORDS):
        return True
    ctxw = Counter()
    for k, c in enumerate(reversed(history[-5:-1])):
        for w in analyze(c)[0]:
            ctxw[w] += 0.7 ** k
    uw = analyze(u)[0]
    if not uw or not ctxw:
        return False
    conn = sum(v for w, v in ctxw.items() if w in uw) / math.sqrt(len(uw)) + \
        sum(v * mat.pmi(x, y) for x, v in ctxw.items() for y in uw) / math.sqrt(len(uw) * sum(ctxw.values()))
    return conn < th


# =============================================================================================
# 5. 指揮役（共感 → 内容 → 質問。課題がありそうなら秘書の型）
# =============================================================================================

EXPERIENCE = re.compile(r"(昔|実家|子供の(頃|時|ころ)|子どもの(頃|時|ころ)|学生の(頃|時|ころ)|(私|わたし|自分)(は|も|自身|の場合)|うちは|私は.{0,12}(ました|てました|ています|てます|でした)|僕|うちの|母が|父が|夫が|妻が|旦那|彼氏|彼女が|始めました|行きました|行ってきました|食べました|してました|してます)")
EMO = {"感動", "嬉しい", "楽しい", "悲しい", "辛い", "大変", "疲れる", "驚く", "困る", "怖い", "寂しい", "面白い", "美味しい", "好き", "嫌い", "心配", "残念", "最高", "幸せ", "ショック", "緊張"}
FUNC = ("助詞", "助動詞", "補助記号", "接尾辞")


class Bot:
    """会話の本体。材料の表・遷移グレイン（非公開の置き場に保存）・キャラクタ・時計・対話者をまとめ、reply(history) で返事を返す。"""

    def __init__(self, char=None, user=None, now=None, rebuild=False, log=print, seed=0, light=False, empathy_plain=0.2):
        self.mat, self.tg = load_tables(rebuild, log)
        self.res = load_light_resonance(self.mat, rebuild, log) if light else self.mat   # 次の発言の共鳴: light = 語板の光（実験09）、既定 = 数え上げの PMI
        self.char = Character(char)
        self.clock = Clock(now)
        self.user = Interlocutor(user) if user else None
        self.user_name = self.user.d.get("名前") if self.user else None
        self.session_new = {}
        self.rng = random.Random(seed)
        self.empathy_plain = empathy_plain                                       # 気持ちの出ていない発言に「〜んですね」を付ける割合（0.3.0、持ち主の評点で多すぎた）

    # ---- 共感: 相手の述語を「〜んですね」で返す（バックトラッキング）----
    def last_predicate(self, ts):
        idx = [i for i, w in enumerate(ts) if w[2] in ("動詞", "形容詞", "形状詞", "名詞")      # 行く・見る も「非自立可能」に入るので除かない
               and not (i > 0 and ts[i - 1][4].startswith("助詞-接続助詞"))]
        emo = [i for i, w in enumerate(ts) if w[1] in EMO]
        return (emo or idx or [None])[-1]

    def argument(self, ts, k):
        """述語のすぐ前の「名詞 + が・を・に・へ・で・と」（何に・何が）。"""
        if k >= 2 and ts[k - 1][4].startswith("助詞-格助詞") and ts[k - 1][0] in ("が", "を", "に", "へ", "で", "と"):
            j = k - 2
            while j >= 0 and ts[j][2] in ("名詞", "接頭辞", "接尾辞") and "副詞可能" not in ts[j][4]:
                j -= 1
            noun = "".join(w[0] for w in ts[j + 1:k - 1])
            if noun and ts[k - 2][1] not in ("私", "僕", "俺", "自分", "何", "事", "物", "こと", "もの", "所", "ところ", "の", "ん"):
                return noun + ts[k - 1][0]
        return ""

    def recover_case(self, verb, history):
        """動詞に「を・に」が無いとき、会話に出た名詞から、格の好みでいちばん合う「名詞 + 格」を補う（強く合う時だけ、実験119）。"""
        ns = [(n, s) for u in history[-4:] for n, s, _ in nouns(u)]
        best = None
        m = self.mat
        for c in ("を", "に"):
            cnt = m.fr.get((verb, c))
            if not cnt:
                continue
            tot = sum(cnt.values())
            for n, s in ns:
                lift = (cnt[n] / tot) / ((m.frn[n] + 1) / m.frt) if cnt[n] >= 2 else 0
                if lift > 5 and (best is None or lift > best[0]):
                    best = (lift, s + c)
        return best[1] if best else ""

    def backtrack(self, user, history=()):
        """「昨日、映画を見てすごく感動したよ」→「感動したんですね」。問い・あいさつには返さない。
        結びの付け替え（Sin-CNN3 実験117）の代わりに、丁寧な形は「原形 + んですね」か、材料から数えた過去の形（見ました → 見た）で返す。"""
        if isq(user) or any(g in user for g in GREET_F):
            return ""
        ts = tokens(user)
        k = self.last_predicate(ts)
        if k is None:
            return ""
        if ts[k][1] in ("為る", "する") and k > 0 and ts[k - 1][2] == "名詞":
            k -= 1
        j = k + 1
        while j < len(ts) and (ts[j][2] in FUNC or ts[j][3] == "非自立可能") and ts[j][2] != "補助記号":
            j += 1
        core, tail = ts[k:k + 1], ts[k + 1:j]
        while tail and (tail[0][3] == "非自立可能" or tail[0][1] in ("為る", "する")):   # 補助の動詞・サ変の「する」は中身に入れる（感動 + し、行っ + て + み …の み）
            core, tail = core + tail[:1], tail[1:]
        tail = [w for w in tail if not w[4].startswith("助詞-終助詞")]
        while tail and tail[-1][1] in ("だ", "です", "の", "ん") and tail[-1][0] != "だっ":
            tail = tail[:-1]
        lem, pos = ts[k][1], ts[k][2]
        arg = self.argument(ts, k)
        if pos == "動詞" and (not arg or arg.endswith(("で", "と"))):
            arg = self.recover_case(lem, list(history)) or arg
        surf = "".join(w[0] for w in core)
        if pos in ("名詞", "形状詞") and not [w for w in tail if w[1] not in ("だ", "です")] and not any(w[1] in ("為る", "する") for w in core[1:]):
            return arg + surf + "なんですね。"
        polite = any(w[1] in ("です", "ます") for w in tail)
        if not polite:
            return arg + surf + "".join(w[0] for w in tail) + "んですね。"
        past = any(w[1] == "た" for w in tail)
        neg = any(w[1] in ("ない", "ん", "ぬ") for w in tail)
        if neg:
            return ""
        head = "".join(w[0] for w in core[:-1])                         # サ変の名詞（感動 + し）など、最後の語の前
        last = core[-1][1]
        if past:
            p = self.mat.past.get(last)
            return arg + head + p + "んですね。" if p else ""
        return arg + head + last + "んですね。"

    # ---- 内容: 返事の候補を選ぶ ----
    def candidates(self, history):
        m, ch = self.mat, self.char
        uw, up = analyze(history[-1])
        s_user = self.tg.state(history[-2] if len(history) >= 2 else "", history[-1])
        ctxw = Counter()
        for k, u in enumerate(reversed(history[-5:-1])):
            for w in analyze(u)[0]:
                ctxw[w] += 0.7 ** k
        if self.user:
            for w, c in sorted(self.user.d["話題"].items(), key=lambda kv: -kv[1])[:10]:
                ctxw[w] += 0.3
        if topic_shift(history, m):
            ctxw = Counter()
        keys = set(uw) | {y for x in uw for y in m.next.get(x, [])[:40]} | {w for w, v in ctxw.most_common(5)}
        cand = Counter()
        for y in keys:
            for k in m.inv.get(y, [])[:3000]:
                cand[k] += 1
        want_polite = ch.polite()
        scored = []
        for k, _ in cand.most_common(400):
            t = m.pool[k]
            if any(t in h for h in history) or any(x in t for x in ch.other_self) or (ch.family_conflict and ch.family_conflict.search(t)):
                continue
            cw_, cp = analyze(t)
            f = (overlap(uw, cw_), self.res.resonance(uw, cw_),
                 1.0 if cp == want_polite else 0.0,                     # 口調（キャラクタの口調と合う候補。光の 2DWIG で置き換える所）
                 self.tg.score(s_user, history[-1], t),
                 float(bool(cw_ & ch.bias)), self.clock.score(t),
                 sum(v for w, v in ctxw.items() if w in cw_ and w not in uw) / math.sqrt(len(cw_) + 1),
                 -float(bool(EXPERIENCE.search(t))),
                 float(isq(t) and bool(re.search(r"(ですか|ますか|ました？|ました\?|でしたか|んですか)", t))))
            scored.append((sum(wi * fi for wi, fi in zip(self.weights(), f)), t, f))
        scored.sort(key=lambda x: -x[0])
        return scored

    def weights(self, jitter=None):
        """手がかりの重み（重なり・共鳴・口調・遷移・興味・時計・前の話題・体験談・相手への問い）。jitter で揺らす（10 通りの票）。"""
        W = self.char.w
        base = [1.0, 1.0, 1.0, 0.5, W.get("興味の語へのバイアス", 0.2), 1.0, 0.5, W.get("体験談を下げる強さ", 0), W.get("相手への問いを上げる強さ", 0)]
        if jitter is None:
            return base
        return [b * (math.exp(jitter.gauss(0, 0.4)) if i not in (5, 7) else 1.0) for i, b in enumerate(base)]

    def ensemble(self, scored, n=10):
        """重みを少しずつ揺らして 10 通り選び、票の多い返事を選ぶ。票の割合 = 自信（持ち主の案）。"""
        rng = random.Random(0)
        votes = Counter()
        for _ in range(n):
            w = self.weights(rng)
            votes[max(scored, key=lambda x: sum(wi * fi for wi, fi in zip(w, x[2])))[1]] += 1
        t, v = votes.most_common(1)[0]
        return t, v / n

    # ---- 質問: 話題を続ける・知らない語を聞き返す・秘書の型 ----
    def related_question(self, user, history):
        m = self.mat
        said = "".join(history)
        ns = sorted(nouns(user), key=lambda x: (not x[2], m.nounf[x[0]]))
        for n, s, prop in ns:
            if not prop:
                continue
            c = m.co.get(n)
            if not c or m.nounf[n] < 5:
                continue
            best = max(((v / math.sqrt(m.nounf[y] + 10), y) for y, v in c.items() if v >= 10 and y not in said and m.nounf[y] >= 5
                        and len(y) >= 2 and y not in STOP and not all("ぁ" <= ch <= "ゖ" for ch in y)), default=None)
            if best:
                return f"{best[1]}ですか？"
        return None

    def unknown_word(self, user):
        m = self.mat
        for n, s, _ in nouns(user):
            if m.cw.get(n, 0) + m.cnext.get(n, 0) == 0 and n not in self.session_new:
                self.session_new[n] = set(analyze(user)[0]) - {n}
                return s
        return None

    @staticmethod
    def secretary_offer(user, bt):
        mm = re.match(r"(.+?)[をにが]", bt) or re.search(r"([一-龥ァ-ヶー]{2,8})(?:が|を)", user)
        noun = mm.group(1) if mm and 1 < len(mm.group(1)) <= 8 else None
        if noun:
            noun = re.sub(r"^(最近|今日|昨日|明日|今週|来週|先週|今)", "", noun) or None
        if re.search(r"たい(んだ|んです|ん)?(けど|が|のに)", user) and noun:
            return f"どんな{noun}がありますか？ よければ一緒に整理しましょう。"
        if noun:
            return f"{noun}のどんなところが一番大変ですか？ 何かお手伝いできることがあれば言ってくださいね。"
        return "どんなところが一番大変ですか？ 何かお手伝いできることがあれば言ってくださいね。"

    def reply(self, history):
        """相手の発言までの会話（古い順）→ (返事, 自信)。"""
        W = self.char.w
        u = history[-1]
        if self.user:
            self.user.update(u)
            if len(history) == 1 and self.user.d.get("名前") and self.user.d.get("会った回数", 0) >= 1 and any(g in u for g in ("こんにちは", "こんばんは", "おはよう", "久しぶり", "また")):
                self.user_name = self.user.d["名前"]
                return self.user.welcome_back(), 1.0
        if len(history) >= 2 and "お名前" in history[-2] and not self.user_name:
            mm = re.match(r"^(?:私は|僕は)?([^、。！!？?\s]{1,8}?)(?:です|と申します|といいます|だよ|と言います)", u)
            if mm:
                self.user_name = mm.group(1)
                if self.user:
                    self.user.d["名前"] = self.user_name
                rest = u[mm.end():].lstrip("。、！! ")
                sa = self.char.answer(rest, history, self.clock) if rest else None
                return f"{self.user_name}さん、よろしくお願いします！" + (sa or ""), 1.0
        sa = self.char.answer(u, history, self.clock)
        if sa:
            return sa, 1.0
        feel = any(e in u for e in EMO_F) or problem_score(u) >= 1.0            # 気持ち・困りごとが出ている発言（共感の「〜んですね」が生きる所）
        p_bt = W.get("共感の強さ", 1.0) if feel else W.get("共感の強さ_ふだん", self.empathy_plain)
        bt = self.backtrack(u, history[:-1]) if self.rng.random() < p_bt else ""
        interest = W.get("課題解決への興味", 0.0)
        if interest > 0 and not isq(u) and problem_score(u) >= 1.0 / interest:
            return bt + self.secretary_offer(u, bt), 1.0
        new = self.unknown_word(u)
        if new and self.rng.random() < W.get("知らない語を聞き返す強さ", 1.0):
            return f"「{new}」って何ですか？ 初めて聞きました。", 0.0
        scored = self.candidates(history)
        if not scored:
            return bt + "もう少し教えてもらえますか？", 0.0
        best, conf = self.ensemble(scored)
        parts_ = [bt]
        if conf < W.get("自信のしきい値", 0.5):
            mm = re.match(r"(.+?)[をにが]", bt)
            if mm and 1 < len(mm.group(1)) <= 8:
                return bt + f"どんな{mm.group(1)}{'でした' if 'た' in bt[mm.end():] else 'です'}か？", conf
            rq = self.related_question(u, history)
            if rq:
                return bt + rq, conf
            parts_.append("（自信がないけど）")
        parts_.append(best)
        if not isq(best) and len(best) < 30 and self.rng.random() < W.get("質問の多さ", 1.0) * W.get("話題を続ける強さ", 1.0):
            rq = self.related_question(u, history + [best])
            if rq:
                parts_.append(rq)
        out = ""
        for p_ in parts_:
            if p_:
                out += ("。" if out and not out.endswith(("。", "！", "？", "?", "!", "）", "w", "笑")) else "") + p_
        return out.replace("？。", "？").replace("！。", "！"), conf


# =============================================================================================
# 6. 読み込みと保存（表は非公開の置き場に。材料が CC BY-SA 4.0 なので、配るときは同じ条件）
# =============================================================================================

def load_tables(rebuild=False, log=print):
    """材料の表と遷移グレインを、非公開の置き場から読む（無ければ作って保存する）。"""
    path = os.path.join(PRIVATE, "models", f"tables_{__version__}.pkl")
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.perf_counter()
    convs = load_conversations()
    log(f"会話 教え {len(convs['train'])}・試し {len(convs['test'])}")
    mat = Material().build(convs, log)
    tg = TransitionGrain().build(convs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((mat, tg), f)
    log(f"表を作った（{time.perf_counter() - t0:.0f} 秒）: {path}")
    return mat, tg


def main(argv=None):
    """手元で話す: python kotoyui.py --chat [--user=名前] [--char=プロフィールの json] [--now=2026-12-24T21:30] [--light（共鳴を語板の光で）]。引数が無ければ決まった例で動かす。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = lambda k: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), None)
    bot = Bot(char=opt("char"), user=opt("user"), now=opt("now"), rebuild="--rebuild" in argv, light="--light" in argv)
    if "--chat" in argv:
        hist = []
        while True:
            try:
                u = input("人: ").strip()
            except EOFError:
                break
            if not u:
                break
            hist.append(u)
            r, sc = bot.reply(hist)
            hist.append(r)
            print("ことゆい:", r, f"（自信 {sc:.1f}）", flush=True)
        if bot.user:
            bot.user.save(len(hist))
        return
    for u in [a for a in argv if not a.startswith("--")] or ["昨日、映画を見てすごく感動したよ", "こんにちは！最近ラーメンにはまってるんだ", "休みの日は何をしていますか？",
                                                             "犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます", "最近寝不足でつらいよ"]:
        r, sc = bot.reply([u])
        print("人:", u)
        print("ことゆい:", r, f"（自信 {sc:.1f}）\n", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, ROOT)
    import kotoyui                                   # 表を pickle で保存するので、クラスは「kotoyui」の名前で作る（__main__ の名前にしない）
    kotoyui.main()
