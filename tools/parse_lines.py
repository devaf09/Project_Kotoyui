"""テキスト（1 行 = 1 文や台詞）を GiNZA にかけ、介入の研究のプログラムが読む解析済みの JSON にする。

使い方: python tools/parse_lines.py <出力.json> <テキスト 1> [<テキスト 2> ...]
  出力は [{name, lines: [{text, toks}]}]。toks の形は intervention/i27f_generate.py の to_toks。
  要るもの: pip install ginza ja_ginza
  元の文をそのまま含むので、出力はリポジトリの外に置く。
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "intervention"))
from i27f_generate import to_toks
import spacy
nlp = spacy.load("ja_ginza")
works = []
for p in sys.argv[2:]:
    lines = [x.strip() for x in open(p, encoding="utf-8") if x.strip()]
    docs = nlp.pipe(lines, batch_size=64)
    works.append(dict(name=os.path.basename(p), lines=[dict(text=x, toks=to_toks(d)) for x, d in zip(lines, docs)]))
    print(p, len(lines), flush=True)
json.dump(works, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
