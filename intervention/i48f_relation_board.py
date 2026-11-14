"""I48f: 語板 v3 の L3「関係」を、保存して読み込める板にする。

I48e（i48e_relation_split.py）で測った形を、そのまま板にした物。中身は 3 つ:
- 語の波（流れから学んだ波）。位相 8 ビットにしてから鍵を作るので、保存した物と計算に使う物が同じになる。
- 語ごとの帯（名詞の後ろ・述語の後ろ、1 本ずつ）。その語の後ろに出たつなぎの印の重ね合わせ = 平均の関係。語板の中に置く物。
- 関係の置き場（名詞の後ろ・述語の後ろ）。場面ごとの鍵の波と、その場面のつなぎ。行で変わる分を受け持つ。波の板の側に置く物。

鍵は 語・品詞 ＋ 印（問い・前の行の問い・否定・既出・先頭・末尾）＋ 係る語。係り方（解析の dep）は入れない
（解析器が助詞から決める物で、答えが漏れる。bot の行づくりでは述語は決まっているが、役は決まっていない）。

標準は D 2048 ＋ 位相 8 ビット（--store p8）。エッジ向けは --D 512。計算は complex64。
資料は --cache で替える。公開の版は RealPersonaChat。非公開の資料で作った板は、名前に _wn を付けて PC の中だけに置く。

作る:   python intervention/i48f_relation_board.py --cache <parsed.json> --out <板の名前> [--D 2048 --store p8 --device cuda:0]
使う:   rb = RelationBoard.load(<板の名前>, device="cuda:0"); rb.top_after("名", "猫", head="歩く")
bot:    RelationLight(rb) は i48c の ParticleLight と同じ口（役らしさ・持ち上げ・助詞）。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402
from i48_wave_board import Store, Waves  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

log = logging.getLogger("i48f")
HOSTS = ("名", "述")
NAMES = {"名": "名詞の後ろ（助詞）", "述": "述語の後ろ（語尾）"}
NO_FLAGS = {f: False for f in E.FLAGS}
VERSION = "v3-L3-1"


class Side:
    """片方の板（名詞の後ろ、か、述語の後ろ）。"""

    def __init__(self, cands, rel):
        self.cands, self.rel = cands, rel
        self.cix = {c: i for i, c in enumerate(cands)}
        self.K = self.Kt = self.P = self.prior = None
        self.band_ix, self.band, self.band_n = {}, None, None


class RelationBoard:
    def __init__(self, D, store, keys, top, device=None):
        self.D, self.store, self.keys, self.top = D, store, tuple(keys), top
        self.device = device if (device and torch is not None) else None
        self.wstore, self.kstore = Store(store, unit=True), Store(store)
        self.side, self.meta = {}, {}

    # ------------------------------------------------------------ 作る
    @classmethod
    def build(cls, works, D=2048, store="p8", keys=("語", "印", "係"), min_count=5, max_scenes=200000, top=30, seed=0, device=None,
              source=""):
        t0 = time.time()
        rb = cls(D, store, keys, top, device)
        vocab = sorted({t["lemma"] for w in works for ln in w["lines"] for t in ln["toks"] if t["pos"] in E.CUE})
        seq = [[t["lemma"] for t in ln["toks"] if t["pos"] in E.CUE] for w in works for ln in w["lines"] + [dict(toks=[])]]
        wv = Waves(vocab, D, seed)
        wv.learn(seq, dev=torch.device(device) if rb.device else None)
        wv.W = rb.wstore.decode_np(rb.wstore.encode(wv.W)).astype(np.complex64)      # しまう形にしてから使う
        wv.base = None
        rb.wv = wv
        S = [s for w in works for s in E.samples(w)]
        for host in HOSTS:
            Sh = [s for s in S if s["host"] == host]
            cnt = Counter(s["post"] for s in Sh)
            cands = [c for c, n in cnt.most_common() if n >= min_count]
            cs = set(cands)
            Sc = [s for s in Sh if s["post"] in cs]
            chosen = E.pick_cues(Sc, "post", 64) if "絞" in keys else []
            sd = Side(cands, E.Relation(wv, cands, D, keys, chosen))
            sd.prior = np.array([cnt[c] for c in cands], float)
            band = {}
            for s in Sc:                                    # 帯は全部の場面から（平均の関係）
                i = wv.ix.get(s["w"])
                if i is not None:
                    if i not in band:
                        band[i] = [np.zeros(D, np.complex64), 0]
                    band[i][0] += sd.rel.cmark[sd.cix[s["post"]]]
                    band[i][1] += 1
            ids = sorted(band)
            sd.band_ix = {i: r for r, i in enumerate(ids)}
            sd.band = rb.kstore.decode_np(rb.kstore.encode(np.stack([band[i][0] / band[i][1] for i in ids])))
            sd.band_n = np.array([band[i][1] for i in ids])
            Sk = Sc[:: max(1, math.ceil(len(Sc) / max_scenes))]      # 置き場は等間隔に間引く
            sd.K = rb.kstore.decode_np(rb.kstore.encode(np.stack([sd.rel.key(s) for s in Sk])))
            sd.P = np.array([sd.cix[s["post"]] for s in Sk], np.int32)
            rb.side[host] = sd
            rb.to_device(host)
            log.info("%s: 候補 %d、帯 %d 語、置き場 %d 場面（%.0f 秒）", NAMES[host], len(cands), len(ids), len(Sk), time.time() - t0)
        rb.meta = dict(version=VERSION, source=source, n_works=len(works), seed=seed, min_count=min_count, built=time.strftime("%Y-%m-%d %H:%M"))
        return rb

    def to_device(self, host):
        sd = self.side[host]
        if self.device:
            sd.Kt = torch.from_numpy(sd.K).to(self.device)
            sd.K = None                                     # GPU に載せた写しを PC のメモリに残さない

    # ------------------------------------------------------------ 保存と読み込み
    def save(self, path):
        arr = dict(W=self.wstore.encode(self.wv.W))
        meta = dict(self.meta, D=self.D, store=self.store, keys=list(self.keys), top=self.top, vocab=self.wv.vocab, sides={})
        for h, sd in self.side.items():
            K = sd.K if sd.K is not None else sd.Kt.cpu().numpy()
            arr.update({f"K_{h}": self.kstore.encode(K), f"P_{h}": sd.P, f"band_{h}": self.kstore.encode(sd.band),
                        f"band_n_{h}": sd.band_n, f"band_ids_{h}": np.array(sorted(sd.band_ix, key=sd.band_ix.get)), f"prior_{h}": sd.prior})
            meta["sides"][h] = dict(cands=sd.cands, chosen=sorted(sd.rel.chosen))
        np.savez(path + ".npz", **arr)
        json.dump(meta, open(path + ".json", "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path, device=None):
        m = json.load(open(path + ".json", encoding="utf-8"))
        z = np.load(path + ".npz")
        rb = cls(m["D"], m["store"], m["keys"], m["top"], device)
        wv = Waves.__new__(Waves)
        wv.vocab, wv.D, wv.spare, wv.base = m["vocab"], m["D"], [], None
        wv.ix = {w: i for i, w in enumerate(wv.vocab)}
        wv.W = rb.wstore.decode_np(z["W"]).astype(np.complex64)
        rb.wv = wv
        for h, sm in m["sides"].items():
            sd = Side(sm["cands"], E.Relation(wv, sm["cands"], rb.D, rb.keys, sm["chosen"]))
            sd.K, sd.P, sd.prior = rb.kstore.decode_np(z[f"K_{h}"]), z[f"P_{h}"], z[f"prior_{h}"]
            sd.band, sd.band_n = rb.kstore.decode_np(z[f"band_{h}"]), z[f"band_n_{h}"]
            sd.band_ix = {int(i): r for r, i in enumerate(z[f"band_ids_{h}"])}
            rb.side[h] = sd
            rb.to_device(h)
        rb.meta = {k: v for k, v in m.items() if k not in ("vocab", "sides")}
        return rb

    # ------------------------------------------------------------ 読む
    @staticmethod
    def scene(w, pos="NOUN", head=E.NONE, dep="?", cues=(), flags=None, nxt=E.NONE, npost=E.NONE):
        return dict(w=w, pos=pos, head=head, dep=dep, cues=list(cues), flags=dict(NO_FLAGS, **(flags or {})), nxt=nxt, npost=npost)

    def scoop(self, host, S):
        """場面の並び → つなぎの明るさ（場面 × 候補）。問いと共鳴の強い top 場面のつなぎを、共鳴の 2 乗で重ねる。"""
        sd = self.side[host]
        Q = np.stack([sd.rel.key(s) for s in S])
        if sd.Kt is not None:
            R = torch.abs(sd.Kt @ torch.from_numpy(np.conj(Q)).to(self.device).T)
            val, o = torch.topk(R, self.top, dim=0)
            val, o = val.cpu().numpy(), o.cpu().numpy()
        else:
            R = np.abs(sd.K @ np.conj(Q).T)
            o = np.argpartition(-R, self.top, axis=0)[: self.top]
            val = np.take_along_axis(R, o, 0)
        out = np.zeros((len(S), len(sd.cands)))
        for c in range(len(S)):
            out[c] = np.bincount(sd.P[o[:, c]], weights=val[:, c] ** 2, minlength=len(sd.cands))
        return out + 1e-6 * sd.prior / sd.prior.sum()

    def band_read(self, host, w):
        """語の帯から読む（平均の関係）。雑音の床より下は 0。帯が無い語は None。"""
        sd = self.side[host]
        r = sd.band_ix.get(self.wv.ix.get(w, -1))
        if r is None:
            return None
        b = sd.band[r]
        v = (sd.rel.cmark @ np.conj(b)).real / self.D
        floor = np.linalg.norm(b) / (self.D * math.sqrt(2)) * math.sqrt(2 * math.log(len(sd.cands)))
        return np.where(v > floor, v, 0.0)

    def top_after(self, host, w, k=5, **kw):
        """語 w の後ろのつなぎを、明るい順に k こ: [(つなぎ, 明るさの割合)]。"""
        sc = self.scoop(host, [self.scene(w, pos="NOUN" if host == "名" else "VERB", **kw)])[0]
        sc = sc / sc.sum()
        return [(self.side[host].cands[i], round(float(sc[i]), 4)) for i in np.argsort(-sc)[:k]]

    def accuracy(self, works, n=4000, seed=1):
        """外した対話で当たりを測る（板の物差し）。"""
        S = [s for w in works for s in E.samples(w)]
        out = {}
        for h, sd in self.side.items():
            te = [s for s in S if s["host"] == h and s["post"] in sd.cix]
            te = random.Random(seed).sample(te, min(n, len(te)))
            SC = np.concatenate([self.scoop(h, te[i:i + 256]) for i in range(0, len(te), 256)])
            r = E.measure(SC, te, sd.cands, h)
            B = np.stack([b if b is not None else sd.prior for b in (self.band_read(h, s["w"]) for s in te)]) + 1e-6 * sd.prior / sd.prior.sum()
            r["band_top1"] = E.measure(B, te, sd.cands, h)["top1"]
            r["n_test"] = len(te)
            out[NAMES[h]] = r
        return out


ROLE_PARTS = {"主語": ("は", "が", "も", "って"), "目的語": ("を",), "補語": ("に", "で", "へ", "と", "から", "まで")}


class RelationLight:
    """bot の行づくりにつなぐ口（i48c_particle_wave.ParticleLight と同じ呼び方）。
    (名詞, 述語) を「名詞の後ろ」の板ですくい、助詞 1 つだけのつなぎの明るさから、役らしさ・持ち上げ・助詞を読む。"""

    HEAD = {"する": "主語", "される": "目的語", "で・に": "補語"}

    def __init__(self, rb):
        self.rb, sd = rb, rb.side["名"]
        self.cands = sd.cands
        self.cls = {r: np.array([c in ps for c in sd.cands]) for r, ps in ROLE_PARTS.items()}
        self.any = np.any(list(self.cls.values()), axis=0)
        tot = sd.prior[self.any].sum()
        self.base = {r: float(sd.prior[m].sum() / tot) for r, m in self.cls.items()}
        self.seen, self.memo = set(), {}

    def set_ctx(self, ctx):
        """会話の前の 3 行の中身の語。既出の印に使う。"""
        self.seen, self.memo = set(ctx), {}

    def prepare(self, pairs):
        """(名詞, 述語) の組をまとめてすくっておく（1 つずつ問うより速い）。"""
        todo = [x for x in dict.fromkeys(pairs) if x not in self.memo]
        for i in range(0, len(todo), 256):
            part = todo[i:i + 256]
            SC = self.rb.scoop("名", [self.rb.scene(n, head=p, flags=dict(既出=n in self.seen)) for n, p in part])
            for x, sc in zip(part, SC):
                sc = np.where(self.any, sc, 0.0)
                self.memo[x] = sc / max(sc.sum(), 1e-12)

    def parts_light(self, n, p):
        if (n, p) not in self.memo:
            self.prepare([(n, p)])
        return self.memo[(n, p)]

    def known(self, n, p):
        return n in self.rb.wv.ix or p in self.rb.wv.ix

    def __call__(self, n, role, p):
        if not self.known(n, p):
            return 1.0
        return min(1.0, float(self.parts_light(n, p)[self.cls[role]].sum()) / max(self.base[role], 1e-9))

    def lifts(self, n, p):
        if not self.known(n, p):
            return None
        pl = self.parts_light(n, p)
        return {h: float(pl[self.cls[r]].sum()) / max(self.base[r], 1e-9) for h, r in self.HEAD.items()}

    def particle(self, n, role, p):
        role = self.HEAD.get(role, role)
        sc = np.where(self.cls[role], self.parts_light(n, p), 0.0)
        return self.cands[int(sc.argmax())] if sc.max() > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="解析済みの会話の資料")
    ap.add_argument("--out", required=True, help="板の名前（.npz と .json を書く）")
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--store", default="p8", choices=Store.NAMES)
    ap.add_argument("--keys", default="語,印,係")
    ap.add_argument("--test_works", type=int, default=500, help="物差し用に外す対話。0 なら全部で作り、測らない")
    ap.add_argument("--max_scenes", type=int, default=200000)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    works = json.load(open(a.cache, encoding="utf-8"))
    works = works["works"] if isinstance(works, dict) else works
    random.Random(a.seed).shuffle(works)
    te, tr = works[: a.test_works], works[a.test_works:]
    rb = RelationBoard.build(tr, a.D, a.store, tuple(a.keys.split(",")), a.min_count, a.max_scenes, seed=a.seed, device=a.device,
                             source=os.path.basename(a.cache))
    if te:
        rb.meta["accuracy"] = rb.accuracy(te)
        for k, r in rb.meta["accuracy"].items():
            log.info("%s: すくう 1 番目 %.3f（3 番まで %.3f）、帯 1 番目 %.3f", k, r["top1"], r["top3"], r["band_top1"])
    rb.save(a.out)
    del rb
    if torch is not None and a.device:
        torch.cuda.empty_cache()
    t0 = time.time()
    rb2 = RelationBoard.load(a.out, a.device)
    log.info("読み込み %.1f 秒、%.2f GB", time.time() - t0, os.path.getsize(a.out + ".npz") / 1e9)
    if te:
        acc = rb2.accuracy(te)
        same = all(acc[k]["top1"] == rb2.meta["accuracy"][k]["top1"] for k in acc)
        log.info("読み込んだ板の当たりは保存前と%s", "同じ" if same else f"違う: {acc}")
    for host, w, kw in (("名", "猫", dict(head="歩く")), ("名", "学校", dict(head="行く")), ("名", "ご飯", dict(head="食べる")),
                        ("述", "歩く", {}), ("述", "食べる", dict(flags=dict(問い=True)))):
        log.info("%s %s %s → %s", host, w, kw, rb2.top_after(host, w, **kw))


if __name__ == "__main__":
    main()
