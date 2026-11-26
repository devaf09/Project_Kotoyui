"""I49zzzb: 会話の行から だれが・何を・どうした を取り、引き継ぎを数える（小さな作り物の会話）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("intervention", "tools"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import i49zzzb_dialogue_slots_run as B


def _line(who, s, o, v):
    toks, i = [], 0
    for lemma, dep in ((s, "nsubj"), (o, "obj")):
        if lemma:
            toks.append(dict(i=i, text=lemma, lemma=lemma, pos="NOUN", dep=dep, head=-1))
            i += 1
    toks.append(dict(i=i, text=v, lemma=v, pos="VERB", dep="ROOT", head=i))
    for t in toks[:-1]:
        t["head"] = i
    return dict(text="x", speaker=who, toks=toks)


def test_slots_reads_subject_object_and_root():
    assert B.slots(_line("a", "私", "本", "読む")["toks"]) == ("私", "本", "読む")
    assert B.slots(_line("a", None, None, "行く")["toks"]) == (None, None, "行く")
    assert B.slots([]) == (None, None, None)


def test_count_pairs_by_speaker():
    work = dict(name="t", lines=[_line("a", "私", "本", "読む"), _line("b", "私", "本", "買う"), _line("a", "私", None, "寝る"), _line("b", None, "本", "買う")])
    d = B.count([work])
    assert d["行の数"] == {"行": 4, "だれが": 3, "何を": 3, "どうした": 4}
    reply = d["組"]["相手の行 → 返事"]
    assert reply["だれが"] == [2, 2] and reply["どうした"] == [3, 0] and reply["何を"] == [1, 1]
    own = d["組"]["同じ話し手の 1 個前の行 → 今の行"]
    assert own["だれが"] == [1, 1] and own["どうした"] == [2, 1]
    assert d["だれがが一人称"] == [3, 3] and d["だれがの分け方"]["続く"] == 2
