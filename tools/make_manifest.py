"""data/MANIFEST.json を作る: データごとの取得元・取得日・ライセンス・条件・ファイルの数と大きさ・指紋（sha256）。

データ自体はリポジトリに入れないので、同じデータを取り直したか確かめられるように、ここに記録する（MANIFEST.json も data/ の下で git の対象外。
中身の要約は README の「データの出典」と NOTES.md に書く）。
実行: python tools/make_manifest.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_ROOT = os.environ.get("KOTOYUI_DATA", os.path.join(ROOT, "..", "Project_Kotoyui_private", "data"))   # 会話のデータは非公開の置き場（2026-11-03 持ち主の依頼、git の外）
DATA = DATA_ROOT

SOURCES = [
    dict(name="aozora", path="aozora/raw/*.zip", retrieved="2026-10-22〜2026-10-24",
         source="青空文庫 https://www.aozora.gr.jp/（公式の作品一覧 list_person_all_extended_utf8.zip の テキストファイル URL）",
         license="著作権の切れた作品（作品著作権フラグ なし）。青空文庫の「収録ファイルの取り扱い規準」により自由に複製・再配布できる",
         note="最初の 1,000 作品は SN2-Parts-2D が取得したものを写した（picked.json）。残りは tools/fetch_aozora.py で取得（picked_all.json）"),
    dict(name="realpersonachat", path="realpersonachat/real_persona_chat/**/*.json", retrieved="2026-10-24",
         source="RealPersonaChat 1.0.0 https://github.com/nu-dialogue/real-persona-chat（コミット 28d0b6b）",
         license="CC BY-SA 4.0",
         note="話者を特定しようとしない。特定の話者へのなりすましに使わない。作ったデータを配るときは CC BY-SA 4.0"),
    dict(name="jmultiwoz", path="jmultiwoz/dataset/JMultiWOZ_1.0.zip", retrieved="2026-11-02",
         source="JMultiWOZ 1.0 https://github.com/nu-dialogue/jmultiwoz（コミット a78dd33）",
         license="CC BY-SA 4.0",
         note="旅行の案内の対話（店員役と客役）。共鳴・状態遷移・格の好みの数え上げに使う（返事の候補には使わない）。作ったデータを配るときは CC BY-SA 4.0"),
    dict(name="nucc", path="nucc/nucc.zip", retrieved="2026-10-24",
         source="名大会話コーパス https://mmsrv.ninjal.ac.jp/nucc/（nucc.zip）",
         license="CC BY-NC-ND 4.0",
         note="非営利。加工したデータは配らない。使うときは文献（藤村ほか 2011 / Fujimura et al. 2012）を明記する"),
]


def digest(files):
    """ファイルの名前と中身から 1 つの sha256 を作る（並びは名前順）。"""
    h = hashlib.sha256()
    for fn in files:
        h.update(os.path.relpath(fn, DATA).replace("\\", "/").encode())
        with open(fn, "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def main():
    out = []
    for s in SOURCES:
        files = sorted(glob.glob(os.path.join(DATA, s["path"]), recursive=True))
        out.append(dict(s, files=len(files), bytes=sum(os.path.getsize(f) for f in files), sha256=digest(files)))
    with open(os.path.join(DATA, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for s in out:
        print(f"{s['name']}: {s['files']} ファイル、{s['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
