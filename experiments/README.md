# experiments/ — 実験のプログラム

- `expNN_名前.py`: 実験のプログラム（NN は 2 桁、このリポジトリで 01 から。枝番は 01b のように）。冒頭の docstring に
  「実験NN: 名前 — 一言」/ 着想 / 方法 / 指標 / 結果（results/expNN.txt、図は tools/fig_expNN.py）/ 実行（コマンド例 1 行）を書く。
  引数は argparse で受け、`--seed --device --out` を必ず持つ。結果は 1 行 JSON で `--out` に追記する
  （引数をすべて・結果・学習する数（光の部分とデジタルの部分を分けて）・時間・版・torch の版）。
- `est_*.py`: GPU を使う前に、数えた表で案の上限を手早く見積もる。
- `sweeps/sweepNN.sh`: そのまま再実行できる条件の一覧。先頭のコメントに「実験NN: 何を比べるか。データ、学習の長さ、seed 数」。
- 分野が増えたら `experiments/<分野>/` に分ける。
- リポジトリ直下で実行する（例: `python experiments/exp01_wig_bag.py --seed 0 --device cuda --out results/exp01.txt`）。
- `exp14`〜`exp22`・`build_persona_pairs.py`・`rerate_owner.py`・`fig_exp18.py`: 人称・主語の実験。材料と結果は公開していない置き場
  （環境変数 `KOTOYUI_PRIVATE`、無ければリポジトリの隣の `Project_Kotoyui_private/`）から読み書きする。
