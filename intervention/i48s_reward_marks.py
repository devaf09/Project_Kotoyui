"""I48s: 会話の報酬（ロードマップ 40）の土台。bot の返事ごとに、相手の次の行などから付けられる「良」の印（ロードマップ 13・14・16・18・37・38）。
数えるだけ（bot には入れない）。手で置く語リストやしきい値はできるだけ避け、置いた物はここに書く。

印（bot の返事 r、相手の前の行 p（r が答えた行）、相手の次の行 n、それより前の会話の行 H）:
  13 貢献感   : n が r の指摘・聞き返しでなく（I35 の cues と同じ見分けを、記録では次の返事の扱いで読む）、
                受けて続けた（受けの板 uptake > 0。I37 の物差し）か、新しい問い（I48e の is_question）。広げた = そのうえで n に会話の新しい中身の語がある。
  38 感謝の喜び: n に感謝の語がある。置いた語（原形）: THANKS。
  14 共鳴     : r の中身の語のうち、p の語と同じ原形ではないが、意味の波が近い語（I48m の Near: |cos| が乱数どうしの床 σ√(2 ln V) を越える）がある。
                そのまま返し = p と同じ原形の語がある。共鳴だけ = 近い語があり、同じ語は無い。
  18 アハ体験 : r が、p の中身の語と、p より前の行（H、話し手を問わない）の中身の語の両方を持つ（別々の行の語を結ぶ）。受けられた = n が 13 の受けた。
  16 好奇心の作り直し: r が問い（is_question）の時、n に会話の新しい中身の語がいくつあるか（相手の答えで板に入る新しい語の数の代わり）。
  37 思いやり : bot の本の物差し（Bot2.oddness: 述語に合わない助詞・裸の名詞。本の数え）で、相手の行に見つけた不自然な所（差 > 0.5。
                chat_bot3 の odd_cut と同じ線）と同じ所を、bot 自身が後で言った割合。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i48e_relation_split as E  # noqa: E402

THANKS = ("ありがとう", "有難う", "有り難う", "助かる", "感謝", "サンキュー")   # 置いた語（原形）。感謝の見分けにだけ使う


def content(toks):
    return [t["lemma"] for t in toks if t["pos"] in E.CONTENT]


def thanks(toks):
    return any(t["lemma"] in THANKS or t["text"].startswith(("ありがと", "助か")) for t in toks)


def is_q(ln):
    return E.is_question(ln)


def resonance(r_toks, p_toks, near):
    """(同じ語がある, 近い語がある（同じ語は除く）)。near は i48m_echo_phrase.Near。"""
    P = set(content(p_toks))
    R = [w for w in content(r_toks)]
    same = any(w in P for w in R)
    close = any(w not in P and any(near.near(w, q) for q in P) for w in R) if P else False
    return same, close


def aha(r_toks, p_toks, H):
    """r が p の語と、p より前の行の語の両方を持つか（p にも前の行にもある語は、前の行の側に数えない）。"""
    P = set(content(p_toks))
    R = set(content(r_toks))
    before = set(w for h in H for w in content(h)) - P
    return bool(R & P) and bool(R & before)


def new_words(n_toks, seen):
    return len(set(content(n_toks)) - seen)


class Odd:
    """Bot2.oddness と同じ物差し（本の (述語, 助詞) と名詞の修飾の数え）を、bot を作らずに使う。"""

    def __init__(self, works):
        import chat_bot2 as C2
        self.C2 = C2
        self.pp = self.pn = self.mod = self.nn = None
        C2.Bot2.odd_boards(self, works)
        self._pp_top = {}
        for (p_, _), v in self.pp.items():
            self._pp_top[p_] = max(self._pp_top.get(p_, 0), v)

    def worst(self, toks):
        return self.C2.Bot2.oddness(self, toks)          # (差, (種類, 所), 鍵)


# ---- I48z: 数えの直し。聞き返しは why の先頭ではなく、相手の行の形（「〜って？」で終わる問いの文）で見る ----
def asked_back(ln):
    """相手の行が、前の返事への聞き返し（「？」で終わる文に「〜って？」がある）か。"""
    import re
    t = (ln or {}).get("text", "")
    return any(re.search(r"って[？?]", x) for x in re.split(r"(?<=[。！!？?])", t) if x.rstrip("」 ").endswith(("？", "?")))


def repair_kept(r):
    """bot の返事が、聞き返しへの言い直し（相手の行に答える返事）のまま言えたか（chat_bot3 の why の経路で読む）。"""
    w = r.get("why", "")
    return w.startswith("相手の聞き返し") or "相手の聞き返し →" in w[:40]


def took_counted(took, n):
    """受けたの数え: 相手の次の行が聞き返しなら受けたに数えない。"""
    return bool(took) and not asked_back(n)


def bind_counted(bind, p):
    """結びの数え: 相手の前の行が聞き返し（返事はその言い直し）なら結びに数えない。"""
    return bool(bind) and not asked_back(p)
