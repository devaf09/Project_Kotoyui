"""前後の文がある文書（KWDLC・京大 Wikipedia 注釈コーパス の生の文）に GiNZA で係り受けを付け、CoNLL-U で書き出す（語板 v2）

parse_tatoeba.py と同じ解析器・同じ形。文書の切れ目は「# doc_id = …」、文の順は「# sent_id = 文書-番号」で残す。
入力: KWDLC / WikipediaAnnotatedCorpus の org/ フォルダ（1 ファイル = 1 文書、「# S-ID」の行の次が文）。
実行: python lexicon/parse_docs.py --org <kwdlc/org> <wac/org> --out docs_ginza.conllu [--n_process 4]
"""
from __future__ import annotations

import argparse
import os

import spacy


def read_docs(roots):
    for root in roots:
        src = os.path.basename(os.path.dirname(os.path.abspath(root)))
        for d, _, fs in sorted(os.walk(root)):
            for f in sorted(fs):
                lines = [l.rstrip("\n") for l in open(os.path.join(d, f), encoding="utf-8")]
                sents = [l.replace("　", " ").strip() for l in lines if l and not l.startswith("#")]
                sents = [s for s in sents if len(s) >= 4]
                if len(sents) >= 2:
                    yield f"{src}-{os.path.splitext(f)[0]}", sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_process", type=int, default=4)
    a = ap.parse_args()
    docs = list(read_docs(a.org))
    rows = [(did, k, s) for did, ss in docs for k, s in enumerate(ss)]
    print("文書", len(docs), "文", len(rows), flush=True)
    nlp = spacy.load("ja_ginza")
    with open(a.out, "w", encoding="utf-8") as f:
        for (did, k, text), doc in zip(rows, nlp.pipe((r[2] for r in rows), batch_size=200, n_process=a.n_process)):
            if k == 0:
                f.write(f"# newdoc id = {did}\n")
            f.write(f"# sent_id = {did}-{k}\n# text = {text}\n")
            for t in doc:
                head = 0 if t.dep_ == "ROOT" else t.head.i + 1
                ne = f"NE={t.ent_type_}" if t.ent_type_ else "_"
                f.write(f"{t.i + 1}\t{t.text}\t{t.lemma_}\t{t.pos_}\t{t.tag_}\t_\t{head}\t{t.dep_.lower() if t.dep_ != 'ROOT' else 'root'}\t_\t{ne}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
