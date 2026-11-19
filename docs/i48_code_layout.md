# I48 のコードの形と整理の段取り

持ち主の不安点 7（コードを整理して分けた方が良いか）への答え。整理の前後で数字が 1 つも変わらない事を、答え合わせの道具（`tools/i48_golden.py`）で確かめながら進める。

## 1. 今の形（2026-11-19）

- 試し 1 つにファイル 1 つ（`intervention/i48ab_*.py` 〜 `i48aw` まで 30 本、約 6,500 行）。新しい試しは、前の試しのクラスを継承して足している。
- 選びの本体の継承は 6 段:

```
AK.Model（i48ak_teach: 学び・照らし・選び・介入の別の板）
 └ AN.QModel（i48an_attr_board: 役ごとの属性の介入の板 Q）
    └ AO.OModel（i48ao_many_particles: Q の合わせ方「中間」）
       └ AP.UModel（i48ap_units: 単位ごとの介入の板 B・C・D）
          ├ AR.RModel（i48ar_superpose: 重ねた板）
          └ AS.SModel（i48as_two_stage: 二段の選び）
             └ AT.DModel（i48at_depth: 単位ごとの深さ）
AK.Model
 └ AW.WModel（i48at_surprise: 重さつきの学び）
    └ AV.VModel（i48av_local_surprise: 帯に寄せる・ならす）
```

- 困る所:
  - 上の段は、下の段の `gap_scores`・`refresh`・`clone` を上書きして点を足している。どの段がどの点を足したかが、読まないと分からない（例: AP.UModel は AK.Model の `gap_scores` を直に呼ぶので、Q の点は入らない）。
  - 効果の薄かった物（先読み、二段、単位の深さ、帯の寄せ替え等）も、継承の途中にいるので外しにくい。
  - `G.OTHER` のように、モジュール全体の切り替えがある。
  - 板・数え・測りの部品が、試しのファイルに散らばっている。

## 2. 目指す形

```
intervention/i48core/
  text      文・要点・間（read_conllu・split・keys_of・items・feats・Surface・Func）
  space     語板の属性の空間と詰め物の中身（AttrSpace・Fill）
  boards    板（PairGrain・Ratio・Whole・Gram・sep_ratio）
  model     選びの本体（Model）。点を足す物は「差し込み」（extras）として持つ
  learn     学び（同じ重さ・重さつき）。重さの決め方は差し込み
  units     介入の単位の板・重ねた板（差し込み）
  measure   測り（当たりと害・と の人と人以外・筋の無い共振・想定外の反応・違和感）
  hands     手で置いている値の一覧
intervention/i48a*_*.py   試し（main と、その試しだけの差し込み）
```

- 選びの本体は 1 つ。点を足す物（Q の板、単位の板、介入の別の板、重ねた板）は、次の 3 つを持つ小さな部品にする。足す・外すは、本体の `extras` に入れる・抜くだけ。
  - `score(m, keys, x, f, v)`: 今の点に足す点
  - `refresh(m)`: 表の作り直し
  - `clone()`: 写し
- 学びの重さ（同じ・1−p・−log p・帯・ならし）も、決め方の部品を渡す形にする（I48at で口は作った: `weight`・`adjust_block`・`add_gap`・`finish_weights`）。
- `OTHER`（その他を選べるか）は、モジュールの切り替えから本体の持ち物にする。

## 3. 段取り

1. **入口をまとめる**（済み）。
   - `i48core` で、今の実装を部品の名前で見せる（中身は動かさない）。
   - 手の値の一覧（`i48core.hands.HANDS`）。
   - 答え合わせの道具（`tools/i48_golden.py`）。
2. **選びの本体に差し込みの口を作る**（済み）。
   - `AK.Model` の `gap_scores`・`refresh`・`clone` に `extras` を通した。
   - 差し込みにした物:
     - Q の板: `AN.QScore`（`init_q` で入る）。
     - 単位の板: `AP.UnitScore`（`init_units` で入り、Q は外す。前と同じ）。
   - 重ねた板（RModel）と二段（SModel）は、`gap_scores` を上書きする代わりに `unit_term` を替える形にした。上書きの `gap_scores` は本体の 1 つだけになった。
   - 答え合わせで 2 回とも「同じ」。
3. **実装を `i48core` へ移す**（一部済み）。試しのファイルは `i48core` から読む形にして、古い名前は残す（古い試しも回る）。
   - 済み: 文・要点・間（`i48core.text`: read_conllu・split・Func・Surface・norm）と、語板の属性の空間（`i48core.space`: AttrSpace）。`i48ab_long_grain` は同じ名前で見せる。答え合わせで「同じ」。
   - まだ: 板（PairGrain・Ratio・Whole・Gram）、選びの本体、学び、測り。読み込みの輪を避けるため、まだ移していない名前は使う時に読む形（`__getattr__`）にしている。
4. **効果の薄い物を外す**。効果の表（`docs/i48_effects.md`）で「効かない・下がった」の物は、差し込みのまま既定では入れない。
5. `OTHER` を本体の持ち物にする（済み）。
   - `m.other = True / False` で決める。None なら `G.OTHER` に従う（前と同じ）。
   - 本体の `refresh` で、板（PairGrain・R1・R2・R3）にも同じ扱いを渡す。
   - 判定は `G.other_ok(obj)` 1 つにまとめた。

- 今の差し込みの形（段 2 の後）:

```
AK.Model.gap_scores: 今の点 ＋ 介入の別の板（Li） ＋ Σ extras の score
  QModel  : extras = [QScore]                    （Q の板）
  UModel  : extras = [UnitScore] → m.unit_term   （単位の板 B・C・D）
  RModel  : unit_term = 重ねた板で当てた単位の点
  SModel  : unit_term = 二段（または全部）の単位の点
```

## 4. 答え合わせ

```
python tools/i48_golden.py --conllu <tatoeba_ginza.conllu> --content <content_boards.json> --func <func_boards.json> --out golden.json   # 整理の前に 1 回
python tools/i48_golden.py --conllu ... --content ... --func ... --check golden.json                                                  # 段ごとに
```

- 学び 5,000 文・試し 200 文。次の 4 つの数え（間・文末・文・助詞ごとの当たりと害・と の例の順位・β）が、小数 1e-9 まで同じなら「同じ」。
  - 土台（AK.Model）
  - 単位（UModel・型 C）
  - 驚き（−log p、筋の無い共振つき）
  - その他（OTHER）
- `tests/` の小さな確かめ（手作りの語板）も、段ごとに全部通す。
