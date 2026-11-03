# 試し方 — 環境の用意から、簡単な試験まで

ことゆいを手元で動かして、光の部品と返事を試すための手順です。所要時間は、データなしの試しで 5 分、会話まで 15 分ほどです。

## 1. 環境

| 項目 | 確かめた環境 | 最低限 |
|---|---|---|
| OS | Windows 11（Git Bash） | Linux・macOS でも動く想定（パスは os.path で組み立てています） |
| Python | 3.14.7 | 3.10 以上 |
| PyTorch | 2.14.1（CUDA 12.6） | 2.0 以上。**CPU だけで動きます** |
| 言葉の解析 | fugashi 1.5.2 + unidic-lite 1.0.8 | requirements.txt |
| GPU | RTX 3060 12GB（実験の学習だけ） | なくてよい（本体・例・テストは CPU） |
| ディスク | | データ一式で約 0.5GB、表・語板で約 0.2GB |

```bash
git clone https://github.com/devaf09/Project_Kotoyui
cd Project_Kotoyui
python -m venv .venv && source .venv/bin/activate      # Windows の PowerShell なら .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

非公開の置き場（会話のデータ・作った表）は、既定でリポジトリの隣の `../Project_Kotoyui_private/` です。別の場所にするときは `KOTOYUI_PRIVATE` を設定します（[data_format.md](data_format.md)）。

## 2. データなしで試す（約 1 分）

```bash
python -m pytest tests            # 8 本のテスト（版の表記・光の部品・語板・キャラクタ）
python examples/quickstart.py     # 言葉の解析・光の語板・キャラクタと時計
```

[examples/quickstart.py](../examples/quickstart.py) がやっていること:

| 段 | 中身 | 見どころ |
|---|---|---|
| 1 | 発言 → 中身の語と、丁寧かどうか | `kotoyui.analyze()` |
| 2 | 小さな手書きの表を、1 語 = 1 台の光の器（語板）に覚えさせる | 光の器が出した値が、表の値とほぼ同じになる（`LightResonance`） |
| 3 | 見本のキャラクタ「言結 となり」と時計グレイン | `Character`・`Clock`（assets/ の表） |
| 4 | 会話（データがあるときだけ） | `Bot.reply()` |

出力の例:

```text
2. 光の器が出した共鳴（表の PMI と比べる）
   犬 → 散歩: 光 1.18（表 1.2）
   雨 → 傘: 光 1.47（表 1.5）
3. 時間帯: 朝 / 季節: 秋
   人「あなたのお名前は？」→ 私は言結となりと申します。言結と呼んでください。
```

## 3. 会話のデータを取る（約 5 分）

```bash
bash tools/fetch_dialogue.sh      # RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）・名大会話コーパス（研究の比べ用）
python tools/make_manifest.py     # 取ったデータの指紋を MANIFEST.json に（任意）
```

どれも取得元のライセンス・条件に従ってください（README の「ライセンスとデータの出典」）。RealPersonaChat は、話者の特定やなりすましに使わない条件があります。

## 4. 話す

```bash
python kotoyui.py                          # 決まった 6 つの例で返事を出す（初回は表を作る、約 2 分）
python kotoyui.py --chat --user=山田       # 手元で話す（空行で終わる）。--user の相手の記憶は users/ に保存
python kotoyui.py --chat --light           # 次の発言の共鳴を光の語板で計算する（初回に語板を作る、CPU で約 7 分）
python kotoyui.py --now=2026-12-24T21:30   # 時刻を決めて試す（あいさつ・季節の語が変わる）
```

Python から:

```python
import kotoyui
bot = kotoyui.Bot(light=True)                       # char="自分のキャラクタ.json" でキャラクタを替えられる
reply, confidence = bot.reply(["犬を飼っているんですが、散歩が大変で"])
```

### 4.1 ブラウザの会話コンソール（グッド・バッドの評点つき）

```bash
python tools/console.py --light          # http://127.0.0.1:8765 をブラウザで開く
```

返事ごとに 👍・👎 を押すと、非公開の置き場の `ratings/ratings.jsonl` に、会話の履歴・返事・評点・自信・設定が 1 行ずつ残ります（手元の PC の中だけで動きます。外の部品は使いません）。
空のまま送ると、会話を最初からやり直します。

### 4.2 ローカル AI（Ollama）と並べる

[Ollama](https://ollama.com/) を入れてモデルを用意すると、`http://localhost:11434` の API に、`tools/ollama_client.py` からつなげます（外の部品は使いません）。

```bash
python tools/ollama_client.py <モデル名> "こんにちは"      # つながるか確かめる
python tools/console.py --ai=<モデル名>                   # ことゆいの返事の下に、ローカル AI の返事も並べる
```

`Ollama(モデル名).rate(履歴, 返事)` で、返事の自然さの評点（1〜5）と理由も聞けます（評価役の補助。持ち主の評点の代わりではありません）。

## 5. 自分で集めた会話を足す

```bash
python tools/import_dialogues.py my_chats.txt --check
python tools/import_dialogues.py my_chats.txt --name mine
python kotoyui.py --rebuild
```

形は [data_format.md](data_format.md) の 2.2。足した会話は返事の候補にもなります。

## 6. 実験を回す（GPU があると速い）

```bash
python experiments/exp09_word_grains.py --mode table --device cpu      # 語板の上限（表だけ、数秒）
python experiments/exp09_word_grains.py --mode light --n 64 --device cpu   # 語板を学習して 10 択（CPU で約 6〜20 分）
python experiments/exp08_resonance_fix.py --input pixel --plates xy --read image --device cuda   # 1 本の光路（GPU で約 11 分）
```

結果は results/ に 1 行の JSON で足されます。実験の一覧と読み方は [experiments_guide.md](experiments_guide.md)、数字の正本は [NOTES.md](../NOTES.md)。

## 7. うまく動かないとき

| 症状 | 原因と対処 |
|---|---|
| `ModuleNotFoundError: fugashi` / 辞書が見つからない | `pip install fugashi unidic-lite` |
| 4 の会話が飛ばされる | 会話のデータが無い。3 を実行。置き場を変えたなら `KOTOYUI_DATA` を確かめる |
| データを足したのに返事が変わらない | 表が古い。`python kotoyui.py --rebuild` |
| CPU の計算なのに GPU を使ってしまう | `CUDA_VISIBLE_DEVICES=` を付けて起動する |
