#!/bin/bash
# 会話のデータを取得する（RealPersonaChat・JMultiWOZ・名大会話コーパス。docs/data_plan.md・docs/data_format.md）。リポジトリ直下で実行: bash tools/fetch_dialogue.sh
# どちらも非公開の置き場 ../Project_Kotoyui_private/data/ に置く（git の外。配らない。2026-11-03 持ち主の依頼で data/ から移した）。
# 取ったら python tools/make_manifest.py で MANIFEST.json を作り直す。置き場は KOTOYUI_DATA で変えられる。
cd "$(dirname "$0")/.."
D="${KOTOYUI_DATA:-../Project_Kotoyui_private/data}"
mkdir -p "$D"
# 1. RealPersonaChat（名古屋大学・京都大学、CC BY-SA 4.0）。2026-10-24 に取得したコミットに固定する
if [ ! -d "$D"/realpersonachat ]; then
  git clone https://github.com/nu-dialogue/real-persona-chat.git "$D"/realpersonachat
  git -C "$D"/realpersonachat checkout 28d0b6b3865b29cabc26c230a2db37cdf315e937
fi
# 2. JMultiWOZ 1.0（名古屋大学ほか、CC BY-SA 4.0）。2026-11-02 に取得したコミットに固定し、データの zip を開く
if [ ! -d "$D"/jmultiwoz ]; then
  git clone https://github.com/nu-dialogue/jmultiwoz.git "$D"/jmultiwoz
  git -C "$D"/jmultiwoz checkout a78dd334b907e12318b76bb23d0a9ed1498b8d15
fi
if [ ! -d "$D"/jmultiwoz/dataset/JMultiWOZ_1.0 ]; then
  (cd "$D"/jmultiwoz/dataset && unzip -q -o JMultiWOZ_1.0.zip)
fi
# 3. 名大会話コーパス（研究の比べ用。ことゆいの本体は使わない）（国立国語研究所、CC BY-NC-ND 4.0）。公式のページの nucc.zip を 1 回だけ
if [ ! -f "$D"/nucc/nucc.zip ]; then
  mkdir -p "$D"/nucc
  curl -sSL -o "$D"/nucc/nucc.zip https://mmsrv.ninjal.ac.jp/nucc/nucc.zip
  (cd "$D"/nucc && unzip -q -o nucc.zip)
fi
