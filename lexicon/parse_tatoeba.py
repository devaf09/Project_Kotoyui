"""Tatoeba の日本語の文に GiNZA（spaCy）で品詞と係り受けを付け、CoNLL-U の形で書き出す（UD Japanese-GSD と同じ形にそろえる）

実行: python lexicon/parse_tatoeba.py --src jpn_sentences.tsv.bz2 --out tatoeba_ginza.conllu [--n_process 4]
"""
from __future__ import annotations

import argparse
import bz2

import spacy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_process", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in bz2.open(a.src, "rt", encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    nlp = spacy.load("ja_ginza")
    with open(a.out, "w", encoding="utf-8") as f:
        for (sid, _, text), doc in zip(rows, nlp.pipe((r[2] for r in rows), batch_size=200, n_process=a.n_process)):
            f.write(f"# sent_id = tatoeba-{sid}\n# text = {text}\n")
            for t in doc:
                head = 0 if t.dep_ == "ROOT" else t.head.i + 1
                f.write(f"{t.i + 1}\t{t.text}\t{t.lemma_}\t{t.pos_}\t{t.tag_}\t_\t{head}\t{t.dep_.lower() if t.dep_ != 'ROOT' else 'root'}\t_\t_\n")
            f.write("\n")


if __name__ == "__main__":
    main()
