"""現代の雑談の読み込み — RealPersonaChat と名大会話コーパスを、同じ形の発言の列にそろえる（2026-10-24、持ち主の許可 D1・D2）

データ（どちらもリポジトリには入れない。docs/data_plan.md・README の「データの出典」）:
    data/realpersonachat/: RealPersonaChat 1.0.0（名古屋大学・京都大学、CC BY-SA 4.0）。1 対 1 のチャット 13,583 対話、話者 233 人（人物像・性別・年代つき）。
        条件: 話者を特定しようとしない。特定の話者へのなりすましに使わない。
    data/nucc/nucc/: 名大会話コーパス（国立国語研究所、CC BY-NC-ND 4.0）。雑談 129 会話の書き起こし。非営利、加工したデータは配らない。
        使うときは文献を明記する: 藤村逸子・大曾美恵子・大島ディヴィッド義和（2011）「会話コーパスの構築によるコミュニケーション研究」。
そろえた形: 1 発言 = {"src", "dialog", "turn", "speaker", "text", "fp"}。fp は一人称（aozora.FIRST で表記をまとめたもの、1 種類だけのとき）。
    speaker はデータの中で人を区別する番号（RealPersonaChat は対話をまたいで同じ人。名大会話コーパスは会話ごとの番号で、会話をまたいで同じ人かは分からない）。
名大会話コーパスの整え方: 「＠」の行（会話の情報）は飛ばす。「F107：」で始まる行で話し手が変わり、名前のない行は前の話し手の続きの発言とする。
    （うん）（ふーん）など丸かっこの中（相手のあいづち）と、＜笑い＞などの山かっこの印は消す。【大学名】などの伏せ字と＊＊＊は「＊」にする。

使い方:
    import dialogue
    us = dialogue.load()                 # 2 つのデータの発言（1 回目はキャッシュ data/cache/modern.json を作る）
    dialogue.speaker_classes(us)         # 話者ごとの一人称の類（12 類。一番多く使った類）
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

import aozora

ROOT = aozora.ROOT
RPC = os.path.join(aozora.DATA_ROOT, "realpersonachat", "real_persona_chat")
NUCC = os.path.join(aozora.DATA_ROOT, "nucc", "nucc")
CACHE = os.path.join(aozora.DATA_ROOT, "cache", "modern.json")


def load_rpc():
    """RealPersonaChat の発言。"""
    out = []
    for fn in sorted(glob.glob(os.path.join(RPC, "dialogues", "*.json"))):
        d = json.load(open(fn, encoding="utf-8"))
        for u in d["utterances"]:
            out.append(dict(src="rpc", dialog=d["dialogue_id"], turn=u["utterance_id"], speaker=u["interlocutor_id"], text=u["text"].strip()))
    return out


def clean_nucc(t):
    t = re.sub(r"（[^（）]*）", "", t)
    t = re.sub(r"＜[^＜＞]*＞", "", t)
    t = re.sub(r"【[^【】]*】", "＊", t)
    t = re.sub(r"＊+", "＊", t)
    return t.strip()


def load_nucc():
    """名大会話コーパスの発言。"""
    out = []
    for fn in sorted(glob.glob(os.path.join(NUCC, "data*.txt"))):
        did = int(re.search(r"(\d+)", os.path.basename(fn)).group(1))
        cur, turn = None, 0
        for line in open(fn, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("＠"):
                continue
            m = re.match(r"^([A-Z]+\d*)：(.*)$", line)
            if m:
                cur, line = m.group(1), m.group(2)
            if cur is None:
                continue
            t = clean_nucc(line)
            if t:
                out.append(dict(src="nucc", dialog=did, turn=turn, speaker=f"{did}-{cur}", text=t))
                turn += 1
    return out


def load():
    """2 つのデータの発言に一人称をつけて返す（キャッシュがあれば読む）。"""
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    us = load_rpc() + load_nucc()
    for u in us:
        u["fp"] = aozora.label(u["text"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(us, f, ensure_ascii=False)
    return us


def speaker_classes(us, min_count=1):
    """話者ごとの一人称の類（12 類）。一人称のついた発言で一番多く使った類。min_count 回より少ない話者は除く。"""
    c = defaultdict(Counter)
    for u in us:
        g = aozora.cls(u.get("fp"))
        if g:
            c[(u["src"], u["speaker"])][g] += 1
    return {k: v.most_common(1)[0][0] for k, v in c.items() if sum(v.values()) >= min_count}


def stats(us):
    """データごとの数（発言・話者・一人称つき・話者の類で覆える発言）。"""
    sc = speaker_classes(us)
    out = {}
    for src in ("rpc", "nucc"):
        xs = [u for u in us if u["src"] == src]
        spk = {u["speaker"] for u in xs}
        out[src] = dict(utterances=len(xs), dialogs=len({u["dialog"] for u in xs}), speakers=len(spk),
                        chars=sum(len(u["text"]) for u in xs),
                        with_first_person=sum(1 for u in xs if aozora.cls(u["fp"])),
                        speakers_with_class=sum(1 for s in spk if (src, s) in sc),
                        covered_by_speaker_class=sum(1 for u in xs if (src, u["speaker"]) in sc),
                        class_of_utterances=dict(Counter(aozora.cls(u["fp"]) for u in xs if aozora.cls(u["fp"])).most_common()),
                        class_of_speakers=dict(Counter(v for (s, _), v in sc.items() if s == src).most_common()),
                        first_person_raw=dict(Counter(u["fp"] for u in xs if u["fp"]).most_common(15)))
    return out


if __name__ == "__main__":
    print(json.dumps(stats(load()), ensure_ascii=False, indent=1))
