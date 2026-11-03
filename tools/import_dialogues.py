"""使う人が自分で集めた会話のファイルを、ことゆいが読める形（own/*.jsonl）に直す

ことゆいは、会話の材料（返事の候補と、数え上げ・光の学習の材料）を KOTOYUI_DATA（既定 ../Project_Kotoyui_private/data）から読みます。
このツールで作った own/<名前>.jsonl は、RealPersonaChat などと並んで「own」の出どころとして読まれ、返事の候補にもなります。
自分で集めた会話は、使ってよい物だけにしてください（他人のチャットの記録などは、相手の許可とサービスの規約を確かめてから）。

読める形（拡張子で決める）:
    .txt    1 行 = 1 発言。空行で会話を区切る。行の頭の「名前: 」「名前：」は話者の印として外す（--keep_speaker で残す）
    .csv    列 conversation_id・text（ほかの列は無視）。同じ conversation_id の行を 1 つの会話として、上から順に並べる
    .jsonl  1 行 = 1 会話。{"turns": ["発言", …]} か {"utterances": [{"text": "発言"}, …]}（RealPersonaChat と同じ形）
    .json   上の会話のリスト、または {会話の番号: 会話} の辞書

出す形（own/<名前>.jsonl）: 1 行 = {"id": 会話の番号, "turns": [発言, …]}。4 発言より短い会話は、ことゆいが読むときに飛ばします。

使い方（リポジトリ直下で）:
    python tools/import_dialogues.py my_chats.txt --name mine          # → ../Project_Kotoyui_private/data/own/mine.jsonl
    python tools/import_dialogues.py my_chats.txt --check              # 書き出さず、会話の数・発言の数・最初の会話だけ見る
足したあと、表を作り直す: python kotoyui.py --rebuild
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import OrderedDict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_ROOT = os.environ.get("KOTOYUI_DATA", os.path.join(ROOT, "..", "Project_Kotoyui_private", "data"))
SPEAKER = re.compile(r"^\s*[^:：\s]{1,20}\s*[:：]\s*")


def turns_of(d):
    """1 会話の辞書・リスト → 発言のリスト。"""
    if isinstance(d, list):
        return [str(t) for t in d]
    if "turns" in d:
        return [t if isinstance(t, str) else t.get("text", t.get("utterance", "")) for t in d["turns"]]
    if "utterances" in d:
        return [u["text"] for u in d["utterances"]]
    raise ValueError("会話に turns も utterances もありません")


def read(path, keep_speaker=False):
    """ファイル → [(会話の番号, [発言, …])]。"""
    ext = os.path.splitext(path)[1].lower()
    out = []
    if ext == ".txt":
        cur = []
        for line in open(path, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    out.append((str(len(out)), cur))
                cur = []
                continue
            cur.append(line.strip() if keep_speaker else SPEAKER.sub("", line).strip())
        if cur:
            out.append((str(len(out)), cur))
    elif ext == ".csv":
        convs = OrderedDict()
        for row in csv.DictReader(open(path, encoding="utf-8", newline="")):
            convs.setdefault(row["conversation_id"], []).append(row["text"])
        out = list(convs.items())
    elif ext == ".jsonl":
        for i, line in enumerate(open(path, encoding="utf-8")):
            if line.strip():
                d = json.loads(line)
                out.append((str(d.get("id", d.get("dialogue_id", i))) if isinstance(d, dict) else str(i), turns_of(d)))
    elif ext == ".json":
        d = json.load(open(path, encoding="utf-8"))
        items = d.items() if isinstance(d, dict) else enumerate(d)
        out = [(str(k), turns_of(v)) for k, v in items]
    else:
        raise ValueError(f"読めない形です: {ext}（.txt・.csv・.jsonl・.json）")
    return [(cid, [t.strip() for t in ts if t and t.strip()]) for cid, ts in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", default="mine", help="書き出すファイルの名前（own/<名前>.jsonl）")
    ap.add_argument("--keep_speaker", action="store_true", help=".txt の行の頭の「名前: 」を外さない")
    ap.add_argument("--check", action="store_true", help="書き出さず、数と最初の会話だけ見る")
    a = ap.parse_args()
    convs = []
    for f in a.files:
        convs += [(f"{os.path.basename(f)}:{cid}", ts) for cid, ts in read(f, a.keep_speaker)]
    n_use = sum(len(ts) >= 4 for _, ts in convs)
    print(f"会話 {len(convs)}（うち 4 発言以上 {n_use}）・発言 {sum(len(ts) for _, ts in convs)}")
    if convs:
        print("最初の会話:", convs[0][1][:6])
    if a.check:
        return
    out = os.path.join(DATA_ROOT, "own", f"{a.name}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for cid, ts in convs:
            f.write(json.dumps({"id": cid, "turns": ts}, ensure_ascii=False) + "\n")
    print("書き出した:", out, "（表を作り直す: python kotoyui.py --rebuild）")


if __name__ == "__main__":
    main()
