"""GitHub から取った会話コーパスを 1 行 1 発言の jsonl.gz にそろえる。
使い方: python -I to_jsonl.py <dl_dir> <out_dir>
列: source, dialogue_id, split, turn, speaker, text, (addressee), (meta)
"""
import json, glob, gzip, os, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
stats = {}

def write(name, rows):
    p = os.path.join(out, f"{name}.jsonl.gz")
    n = 0; dl = set(); sp = set(); ch = 0
    with gzip.open(p, "wt", encoding="utf-8") as f:
        for r in rows:
            if not r["text"].strip():
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1; dl.add(r["dialogue_id"]); sp.add((r["dialogue_id"] if r["source"] != "mrmp" else "", r["speaker"])); ch += len(r["text"])
    stats[name] = {"utterances": n, "dialogues": len(dl), "chars": ch, "bytes": os.path.getsize(p)}

# MRMP（多人数・関係つき雑談）
def mrmp():
    base = f"{src}/mrmp/multi_relational_multi_party_chat_corpus"
    for fn in sorted(glob.glob(f"{base}/dialogues/*/*.json")):
        d = json.load(open(fn, encoding="utf-8"))
        for u in d["utterances"]:
            yield {"source": "mrmp", "dialogue_id": d["dialogue_id"], "split": None, "turn": u["utterance_id"],
                   "speaker": u["interlocutor_id"], "text": u["text"], "addressee": u.get("mention_to") or [],
                   "meta": {"type": d["dialogue_type"], "relationship": d["relationship"], "interlocutors": d["interlocutors"]}}
write("mrmp", mrmp())
# 話者の人物像は別ファイル
inter = json.load(open(f"{src}/mrmp/multi_relational_multi_party_chat_corpus/interlocutors.json", encoding="utf-8"))
with gzip.open(os.path.join(out, "mrmp_interlocutors.json.gz"), "wt", encoding="utf-8") as f:
    json.dump(inter, f, ensure_ascii=False)

# JMRD（映画のおすすめ対話）
def jmrd():
    for sp in ["train", "valid", "test"]:
        for d in json.load(open(f"{src}/jmrd/{sp}.json", encoding="utf-8")):
            for t, u in enumerate(d["dialog"]):
                yield {"source": "jmrd", "dialogue_id": d["dialog_id"], "split": sp, "turn": t, "speaker": u["speaker"],
                       "text": u["text"], "meta": {"movie": d["movie_title"]}}
write("jmrd", jmrd())

# RecomMind（映画のおすすめ対話・相手の内面のラベルつき）
def recom():
    for fn in sorted(glob.glob(f"{src}/RecomMind/dialogues/*/*.json")):
        sp = fn.split("/")[-2]
        d = json.load(open(fn, encoding="utf-8"))
        for t, u in enumerate(d["utterances"]):
            yield {"source": "recommind", "dialogue_id": d["dialog_id"], "split": sp, "turn": t, "speaker": u["speaker"],
                   "text": u["text"], "meta": {"speaker_name": d.get(u["speaker"], {}).get("name")}}
write("recommind", recom())

# BSD（ビジネスの場面の対話。日本語側だけ。CC BY-NC-SA）
def bsd():
    for sp in ["train", "dev", "test"]:
        for d in json.load(open(f"{src}/BSD/{sp}.json", encoding="utf-8")):
            for u in d["conversation"]:
                yield {"source": "bsd", "dialogue_id": d["id"], "split": sp, "turn": u["no"], "speaker": u["ja_speaker"],
                       "text": u["ja_sentence"], "meta": {"scene": d["tag"], "title": d["title"], "orig_lang": d["original_language"], "en": u["en_sentence"]}}
write("bsd", bsd())
json.dump(stats, open(os.path.join(out, "stats.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
