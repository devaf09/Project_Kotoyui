"""試しの形の一覧: 判定（docs/i48_effects.md）と、既定で入るか、入れ方。

本体（i48core.model.Model）に既定で入っている物は「既定」が True の物だけ。ほかは、使う時に差し込み・型・試しのファイルで入れる。
効かなかった・下がった物は既定では入れない（外さずに残す。必要な時に入れられる）。
"""

EXPERIMENTS = [
    # 名前, 節, 判定, 既定, 入れ方
    ("節ごとの意味の印（シフト＋元）", "4.30", "効いた", True, "Model の特徴（MODE）"),
    ("三つの共振 R1・R2・R3", "4.35", "効かない", True, "Model の選び（今×三つ）。光の器の部品なので外さない（不安点 1 で効き方を変える）"),
    ("prior の強さを証拠の段ごとに", "4.29", "少し", True, "Model.fit の best_w"),
    ("介入の別の板（sep）", "4.38", "少し", True, "Model.sep = True で教えた分が Li に入る（教えなければ点は 0）"),
    ("暗い側（外した選びを下げる）", "4.37", "効かない", False, "Model.teach で選びが外れた時だけ入る（dark_on）"),
    ("Q の板（役ごとの属性）", "4.41", "少し", False, "QModel.init_q（差し込み QScore）"),
    ("単位の板 B・C・D", "4.43", "少し", False, "UModel.init_units（差し込み UnitScore）。C が良い"),
    ("機械の単位 1 万を外付けで", "4.46", "効いた", False, "SModel で全部 B ＋ 学びに使っていない文から teach_unit"),
    ("重ねた板・二段", "4.44・4.46", "道具（二段は枚数が増えると下がる）", False, "RModel・SModel の unit_term"),
    ("同時に照らす", "4.45", "道具", False, "i48aq_simultaneous.run_simul"),
    ("驚きの重さ −log p", "4.47・4.51", "効いた（23 万では小さい）", False, "i48core.learn.WModel.fit_weighted(..., \"-logp\")"),
    ("単位ごとの深さ・驚き", "4.47 (一)", "効かない", False, "i48at_depth.DModel"),
    ("違和感の順・二回り目", "4.49", "道具", False, "i48at_recheck.recheck・oddness"),
    ("その他を選べる", "4.50", "少し", False, "Model.other = True"),
    ("驚きを帯に寄せる・ならす", "4.52・4.53", "下がった", False, "i48av_local_surprise.VModel"),
    ("先読みの幅を左の整合で絞る", "4.31", "下がった", False, "i48af_beam"),
    ("OK を覚える", "4.34", "効かない", False, "i48ai_remember"),
    ("述語の語板で教える重み", "4.40", "効かない", False, "i48am_attr_weight"),
]


def on_by_default():
    return [e[0] for e in EXPERIMENTS if e[3]]
