"""青空文庫の小説を取得する（持ち主の許可 D3、2026-10-24。docs/data_plan.md）

選び方: 公式の作品一覧 data/aozora/list.zip（list_person_all_extended_utf8.zip）から、分類番号に 913（日本の小説）を含み、
        文字遣いが新字新仮名で、作品著作権フラグが「なし」の作品。作家の没年の新しい順に取る（なるべく新しめの言葉から、持ち主の判断）。
        --min_death より前に亡くなった作家の作品は取らない（0 なら全部）。すでに data/aozora/raw/ にある作品は飛ばす（止まっても続きから）。
取り方: 一覧の「テキストファイル URL」の zip を 1 つずつ、--wait 秒あけて取る（青空文庫のサーバーに負担をかけないため）。
記録: 取った作品の一覧を data/aozora/picked_all.json に書く（id・題・作家・没年・URL）。いままでの 1,000 作品（picked.json）はそのまま残す。
ライセンス: 著作権の切れた作品。青空文庫の「収録ファイルの取り扱い規準」により自由に複製・再配布できる。
実行: python tools/fetch_aozora.py --min_death 1940 --wait 3
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_ROOT = os.environ.get("KOTOYUI_DATA", os.path.join(ROOT, "..", "Project_Kotoyui_private", "data"))   # 会話のデータは非公開の置き場（2026-11-03 持ち主の依頼、git の外）
DATA = os.path.join(DATA_ROOT, "aozora")


def candidates(min_death=0):
    """取る候補の作品（没年の新しい順）。"""
    z = zipfile.ZipFile(os.path.join(DATA, "list.zip"))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8-sig")))
    h = rows[0]
    col = {k: h.index(k) for k in ("作品ID", "作品名", "分類番号", "文字遣い種別", "作品著作権フラグ", "姓", "名", "没年月日", "テキストファイルURL")}
    seen, out = set(), []
    for r in rows[1:]:
        wid = r[col["作品ID"]]
        if wid in seen:
            continue
        seen.add(wid)
        if "913" not in r[col["分類番号"]] or r[col["文字遣い種別"]] != "新字新仮名" or r[col["作品著作権フラグ"]] != "なし":
            continue
        url = r[col["テキストファイルURL"]]
        if not url.endswith(".zip"):
            continue
        m = re.match(r"(\d{4})", r[col["没年月日"]])
        death = int(m.group(1)) if m else 0
        if death < min_death:
            continue
        out.append(dict(id=wid, title=r[col["作品名"]], author=r[col["姓"]] + r[col["名"]], death=death,
                        url=url.replace("http://", "https://", 1)))
    out.sort(key=lambda w: (-w["death"], w["id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_death", type=int, default=1940)
    ap.add_argument("--wait", type=float, default=3.0, help="1 作品ごとに待つ秒数")
    ap.add_argument("--limit", type=int, default=0, help="取る数の上限（0 = 全部）")
    a = ap.parse_args()
    os.makedirs(os.path.join(DATA, "raw"), exist_ok=True)
    ws = candidates(a.min_death)
    if a.limit:
        ws = ws[:a.limit]
    log = os.path.join(DATA, "picked_all.json")
    done = {w["id"]: w for w in json.load(open(log, encoding="utf-8"))} if os.path.exists(log) else {}
    n_new = n_fail = 0
    print(f"候補 {len(ws)} 作品（没年 {a.min_death} 年以降）", flush=True)
    for i, w in enumerate(ws):
        fn = os.path.join(DATA, "raw", f"{w['id']}.zip")
        if not os.path.exists(fn):
            try:
                req = urllib.request.Request(w["url"], headers={"User-Agent": "Project_Kotoyui research (polite fetch, 1 file per few seconds)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                zipfile.ZipFile(io.BytesIO(body))   # 壊れた zip は保存しない
                with open(fn + ".part", "wb") as f:
                    f.write(body)
                os.replace(fn + ".part", fn)
                n_new += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"失敗 {w['id']} {w['url']}: {e}", flush=True)
                time.sleep(a.wait)
                continue
            time.sleep(a.wait)
        done[w["id"]] = w
        if (i + 1) % 50 == 0 or i + 1 == len(ws):
            with open(log, "w", encoding="utf-8") as f:
                json.dump(list(done.values()), f, ensure_ascii=False, indent=0)
            print(f"{i + 1}/{len(ws)} 新しく {n_new}・失敗 {n_fail}", flush=True)
    print(f"終わり: 新しく {n_new}・失敗 {n_fail}・一覧 {len(done)} 作品", flush=True)


if __name__ == "__main__":
    sys.exit(main())
