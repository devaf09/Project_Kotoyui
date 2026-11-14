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


def test_particle_light_roles():
    def ln(words):      # (表記, 品詞, 係り先, 係り) の並び → 解析の形
        return dict(text="".join(w for w, *_ in words), toks=[dict(i=i, text=w, lemma=w, pos=p, head=h, dep=d) for i, (w, p, h, d) in enumerate(words)])
    a = ln([("海", "NOUN", 2, "obl"), ("に", "ADP", 0, "case"), ("行く", "VERB", 2, "ROOT")])
    b = ln([("私", "PRON", 2, "nsubj"), ("は", "ADP", 0, "case"), ("行く", "VERB", 2, "ROOT")])
    c = ln([("パン", "NOUN", 2, "obj"), ("を", "ADP", 0, "case"), ("食べる", "VERB", 2, "ROOT")])
    works = [dict(name=f"w{k}", lines=[a, b, c]) for k in range(4)]
    pl = m.ParticleLight(works, D=256, top=4)
    assert pl("海", "補語", "行く") > pl("海", "主語", "行く")
    assert pl("私", "主語", "行く") > pl("私", "補語", "行く")
    assert pl.particle("パン", "目的語", "食べる") == "を"
    assert pl("知らない", "主語", "知らない") == 1.0


def test_move_roles_keeps_total():
    import i27s_fused_generate as S
    from board_ops import Board
    BB = Board()
    BB[("板", "海", "する:行く")] = 2.0
    BB[("板", "海", "修飾:青い")] = 1.0
    S.role_move = lambda n, p: {"する": 0.1, "される": 0.1, "で・に": 3.0}
    try:
        out = S.move_roles(BB)
    finally:
        S.role_move = None
    tot = sum(out[("板", "海", r + ":行く")] for r in S.MOVE_ROLES)
    assert abs(tot - 2.0) < 1e-9
    assert out[("板", "海", "で・に:行く")] > out[("板", "海", "する:行く")]
    assert out[("板", "海", "修飾:青い")] == 1.0
