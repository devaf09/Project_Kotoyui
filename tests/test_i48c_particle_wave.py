"""I48c: 助詞を波ですくう形が、決まった組（名詞 × 係る語 → 助詞）を読み戻せること。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "intervention"))
import i48c_particle_wave as m  # noqa: E402


def test_scoop_reads_back():
    rule = {("海", "行く"): "に", ("パン", "食べる"): "を", ("空", "きれい"): "が", ("友達", "行く"): "と"}
    tr = [dict(n=n, h=h, p=p, ctx=[], before=[], sib=[], aft=["品:VERB"]) for (n, h), p in rule.items() for _ in range(3)]
    parts = sorted(set(rule.values()))
    wv = m.Waves(sorted({x for k in rule for x in k}), 256, 0)
    ps = m.ParticleScoop(m.ParticleBoard(wv, parts, 256, 0), top=3).fit(tr)
    for (n, h), p in rule.items():
        sc = ps.scores(dict(n=n, h=h, p=p, ctx=[], before=[], sib=[], aft=["品:VERB"]))
        assert parts[int(sc.argmax())] == p
