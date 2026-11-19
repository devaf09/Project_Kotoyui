"""I48 の芯（整理の 1 段目: 入口をひとつにまとめた物）。

text と space は実装をここへ移した（古いファイルは同じ名前で見せる）。ほかは、まだ各試しのファイルの実装を部品ごとの名前で見せるだけ。
段取りは docs/i48_code_layout.md。

部品:
  text    文・要点・間（read_conllu・split・keys_of・items・feats・Surface・Func・made_items）
  space   語板の属性の空間と詰め物の中身（AttrSpace・Fill・ev_bin）
  boards  板（PairGrain・Ratio・Whole・Gram・sep_ratio）
  model   選びの本体（Model: 学び・照らし・選び・介入の板）
  learn   重さつきの学び（WModel: 同じ・1−p・−log p、差し込みの口 weight・adjust_block・add_gap・finish_weights）
  units   介入の単位の板（UModel・lookup・pack）、重ねた板（Stack・Shift）
  measure 測り（measure・と の人と人以外・筋の無い共振・想定外の反応・違和感）
  hands   手で置いている値の一覧（HANDS）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
