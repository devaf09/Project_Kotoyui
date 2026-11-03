"""ことゆいを試す最小のコード例（リポジトリ直下で: python examples/quickstart.py）

1〜3 は会話のデータが無くても動きます（CPU で 10 秒ほど）。4 は会話のデータを取ってから（docs/quickstart.md の 3）。
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import kotoyui  # noqa: E402

print("ことゆい", kotoyui.__version__)

# 1. 言葉の解析: 発言 → 中身の語（原形）と、丁寧かどうか（fugashi + unidic-lite）
words, polite = kotoyui.analyze("犬を飼っているんですが、散歩が大変で")
print("\n1. 中身の語:", sorted(words), " 丁寧:", polite)

# 2. 光の語板: 「この語の次に来やすい語」の小さな表を、1 語 = 1 台の光の器に覚えさせる（実験09 の形）
class ToyTable:
    """数えた表の代わりの、小さな手書きの表（語の組 → 回数と PMI）。"""
    cpair = {("犬", "散歩"): 9, ("犬", "かわいい"): 3, ("雨", "傘"): 8, ("雨", "散歩"): 2}
    pmis = {("犬", "散歩"): 1.2, ("犬", "かわいい"): 0.8, ("雨", "傘"): 1.5, ("雨", "散歩"): 0.3}

    def pmi(self, x, y):
        return self.pmis[(x, y)]


light = kotoyui.LightResonance().build(ToyTable(), n=16, plates=8, steps=300, log=lambda *_: None)
print("\n2. 光の器が出した共鳴（表の PMI と比べる）")
for (x, y), p in ToyTable.pmis.items():
    print(f"   {x} → {y}: 光 {light.lift(x)[y]:.2f}（表 {p}）")
print("   「雨」の後に「傘」の返事:", round(light.resonance({"雨"}, {"傘"}), 2), " 「かわいい」の返事:", light.resonance({"雨"}, {"かわいい"}))

# 3. キャラクタと時計（assets/ の見本「言結 となり」）
clock = kotoyui.Clock("2026-11-03T09:00")
char = kotoyui.Character()
print("\n3. 時間帯:", clock.part()["名前"], "/ 季節:", clock.season())
print("   人「あなたのお名前は？」→", char.answer("あなたのお名前は？", ["x", "y"], clock))

# 4. 会話（会話のデータが要る。無ければ飛ばす）
if os.path.isdir(kotoyui.RPC_DIR):
    bot = kotoyui.Bot(now="2026-11-03T15:00")          # 初回は表を作る（約 2 分）。light=True で共鳴を光の語板に（初回 約 7 分）
    for u in ["犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます"]:
        r, sc = bot.reply([u])
        print("\n4. 人:", u, "\n   ことゆい:", r, f"（自信 {sc:.1f}）")
else:
    print("\n4. 会話のデータが無いので飛ばしました。取り方: bash tools/fetch_dialogue.sh（docs/quickstart.md）")
