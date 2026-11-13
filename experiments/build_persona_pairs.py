"""実験14 の材料（非公開）: 前の発言 → 返事 の組に、返事の話し手の一人称の類を付ける

材料（実験だけ。公開しない。2026-11-03 持ち主の指示で別の試みの資料を流用）:
  rpc   RealPersonaChat（Project_Kotoyui の load_conversations、会話ごとの教え・試しの分け方もそのまま）
  web   持ち主が集めた Web 小説（非公開の置き場の webnovels2.jsonl、環境変数 KOTOYUI_WEBNOVELS で場所を変えられる）。1 話の中で「」の行が続く所を会話とみなす。
        作品ごとに教え・試しを分ける（crc32(作品) % 10 == 0 が試し）。R15 の話は重み 0.1（Sin-CNN3 の約束）
一人称の類: 返事の中の一人称の表記を、後ろに助詞などが続く所だけ数える（私・僕・俺・あたし・わし。2 つ以上の類が出る返事は類なし）。
出力: data/persona/pairs.jsonl（1 行 = {"src", "split", "w", "prev", "reply", "cls"}）
実行（Project_Kotoyui の直下で）: CUDA_VISIBLE_DEVICES= python experiments/build_persona_pairs.py
"""
import json
import os
import re
import sys
import zlib

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
KOTO = KOTO_ROOT
sys.path.insert(0, KOTO)
import kotoyui as K  # noqa: E402

WEB = os.environ.get("KOTOYUI_WEBNOVELS", os.path.join(PRIV, "data", "text", "webnovels2.jsonl"))   # 持ち主が集めた資料（公開しない）
OUT = os.path.join(PRIV, "data", "persona", "pairs.jsonl")
TAIL = r"(?=は|が|の|も|を|に|と|って|、|。|！|？|だ|です|じゃ|なら|ら|達|たち|自身|$)"
CLS = {"私": r"(私|わたし)", "僕": r"(僕|ぼく|ボク)", "俺": r"(俺|おれ|オレ)", "あたし": r"(あたし|アタシ)", "わし": r"(わし|儂)"}
PAT = {c: re.compile(p + TAIL) for c, p in CLS.items()}


def cls_of(t):
    hit = [c for c, p in PAT.items() if p.search(t)]
    if "私" in hit and "あたし" in hit:                 # 「あたし」の中の「たし」は私に当たらないが、念のため
        hit.remove("私")
    return hit[0] if len(hit) == 1 else None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = {}
    with open(OUT, "w", encoding="utf-8") as fo:
        def put(src, split, w, prev, reply):
            c = cls_of(reply)
            fo.write(json.dumps(dict(src=src, split=split, w=w, prev=prev, reply=reply, cls=c), ensure_ascii=False) + "\n")
            n[(src, split, c)] = n.get((src, split, c), 0) + 1
        convs = K.load_conversations()
        for split in ("train", "test"):
            for src, turns in convs[split]:
                if src == "rpc":
                    for a, b in zip(turns, turns[1:]):
                        put("rpc", split, 1.0, a, b)
        for line in open(WEB, encoding="utf-8"):
            d = json.loads(line)
            split = "test" if zlib.crc32(d["work"].encode()) % 10 == 0 else "train"
            w = 0.1 if d.get("r15") else 1.0
            prev = None
            for l in d["text"].split("\n"):
                m = re.fullmatch(r"「(.{2,120})」", l.strip())
                if m:
                    t = m.group(1).strip()
                    if prev is not None:
                        put("web", split, w, prev, t)
                    prev = t
                else:
                    prev = None                                  # 地の文で会話が途切れる
    for k in sorted(n, key=str):
        print(k, n[k])


if __name__ == "__main__":
    main()
