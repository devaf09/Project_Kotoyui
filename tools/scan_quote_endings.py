"""Scan a text corpus's quotes (「…」) for にゃ/ニャ sentence endings + quote ending stats.

Read-only over the corpus. Outputs go next to this script.
Usage: python tools/scan_quote_endings.py <corpus_dir> [out_dir]

corpus_dir holds index.tsv (work_id, work_title, episode_id columns) and <work_id>/<episode_id>.txt.
The outputs quote lines of the corpus, so keep them outside the repository.
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    sys.exit(__doc__)
CORPUS = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()

QUOTE_RE = re.compile(r"「([^「」]*)」")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])")
STRIP_CHARS = "。！？!?…‥、ー〜～♪っッ・ぁァあアぃィ"
FIRST_PERSON = ["俺", "おれ", "オレ", "私", "わたし", "ワタシ", "僕", "ぼく", "ボク",
                "あたし", "アタシ", "わたくし", "うち", "拙者", "我", "われ", "吾輩", "儂", "わし"]
# longest first so わたくし isn't also counted as わた..., あたし not as たし etc.
FP_RE = re.compile("|".join(sorted(map(re.escape, FIRST_PERSON), key=len, reverse=True)))


def strip_tail(s):
    s = s.rstrip()
    while s and (s[-1] in STRIP_CHARS or s[-1].isspace()):
        s = s[:-1]
    return s


def read_text(path):
    with open(path, "rb") as f:
        b = f.read()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace")


def main():
    titles = {}
    episodes = []
    with open(os.path.join(CORPUS, "index.tsv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            titles[row["work_id"]] = row["work_title"]
            episodes.append((row["work_id"], row["episode_id"]))

    n_quotes = n_sents = n_eps = n_missing = 0
    nya_count = Counter()
    stats = defaultdict(Counter)  # work_id -> Counter[(kind, key)]

    with open(os.path.join(OUT, "nya_endings.tsv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["work_id", "episode_id", "work_title", "line_no", "sentence_tail", "line"])
        for work_id, ep_id in episodes:
            path = os.path.join(CORPUS, work_id, ep_id + ".txt")
            if not os.path.exists(path):
                n_missing += 1
                continue
            n_eps += 1
            st = stats[work_id]
            for line_no, line in enumerate(read_text(path).splitlines(), 1):
                for m in QUOTE_RE.finditer(line):
                    n_quotes += 1
                    body = m.group(1)
                    for fp in FP_RE.findall(body):
                        st[("一", fp)] += 1
                    for sent in SENT_SPLIT_RE.split(body):
                        core = strip_tail(sent)
                        if not core:
                            continue
                        n_sents += 1
                        st[("尾1", core[-1:])] += 1
                        if len(core) >= 2:
                            st[("尾2", core[-2:])] += 1
                        last4 = core[-4:]
                        if "にゃ" in last4 or "ニャ" in last4:
                            nya_count[work_id] += 1
                            w.writerow([work_id, ep_id, titles.get(work_id, ""), line_no,
                                        sent.strip()[-15:], line.replace("\t", " ")])

    with open(os.path.join(OUT, "nya_summary.tsv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["work_id", "work_title", "nya_sentences"])
        for wid, c in nya_count.most_common():
            w.writerow([wid, titles.get(wid, ""), c])

    with open(os.path.join(OUT, "quote_ending_stats.tsv"), "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["work_id", "kind", "key", "count"])
        for wid in sorted(stats):
            for (kind, key), c in sorted(stats[wid].items(), key=lambda kv: (kv[0][0], -kv[1])):
                w.writerow([wid, kind, key, c])

    print(f"episodes_read={n_eps} missing={n_missing} works={len(stats)}")
    print(f"quotes={n_quotes} quote_sentences={n_sents}")
    print(f"nya_sentences={sum(nya_count.values())} works_with_nya={len(nya_count)}")


if __name__ == "__main__":
    main()
