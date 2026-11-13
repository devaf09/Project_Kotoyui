"""Hugging Face から日本語の LLM 用データを取る（PC で実行。クラウドからは HF に届かない）。
使い方:
  pip install -U huggingface_hub
  python fetch_hf.py --out <非公開の置き場>/data/hf         # 小さい物（合計 約 3 GB）
  python fetch_hf.py --out ... --fineweb-files 2                                   # Web 文（FineWeb2-edu 日本語）も 2 ファイルだけ
  python fetch_hf.py --out ... --only oasst2_ja dolly_ja                           # 一部だけ
取ったら <out>/MANIFEST_hf.json に 取得元・取得日・ライセンス・リビジョンを書く。
"""
import argparse, json, os, datetime
from huggingface_hub import snapshot_download, HfApi

SETS = {
    # 名前: (repo_id, ライセンス, 中身, 取る範囲 allow_patterns or None)
    "oasst2_ja":      ("llm-jp/oasst2-33k-ja", "Apache-2.0", "人と助手の多ターン会話 32,702（英語 oasst2 を DeepL で翻訳）", None),
    "oasst1_ja":      ("llm-jp/oasst1-21k-ja", "Apache-2.0", "人と助手の会話 21,164（oasst1 の翻訳）", None),
    "oasst1_chat_ja": ("kunishou/oasst1-chat-44k-ja", "Apache-2.0", "oasst1 翻訳の多ターン会話形式 44,042", None),
    "dolly_ja":       ("kunishou/databricks-dolly-15k-ja", "CC BY-SA 3.0", "指示と答え 15,015（翻訳）", None),
    "llm_jp_dataset": ("izumi-lab/llm-japanese-dataset", "CC BY-SA 4.0", "日本語の指示・問答 907 万件（2.4 GB）", None),
    "aozora_clean":   ("globis-university/aozorabunko-clean", "CC BY 4.0", "青空文庫 16,951 作品の本文（241 MB）", None),
}
FINEWEB = ("hotchpotch/fineweb-2-edu-japanese", "ODC-By 1.0（Common Crawl の利用規約にも従う）", "教育的な質で絞った日本語 Web 文（全 518 GB。先頭から N ファイルだけ）")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--only", nargs="*")
ap.add_argument("--fineweb-files", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
api = HfApi()
mpath = os.path.join(a.out, "MANIFEST_hf.json")
man = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else {}
today = datetime.date.today().isoformat()

def get(name, repo, lic, desc, pats):
    info = api.dataset_info(repo)
    d = snapshot_download(repo, repo_type="dataset", local_dir=os.path.join(a.out, name), allow_patterns=pats)
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    man[name] = {"repo": f"https://huggingface.co/datasets/{repo}", "revision": info.sha, "license": lic,
                 "contents": desc, "fetched": today, "bytes": size, "files": pats or "all"}
    json.dump(man, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ok {name}: {size/1e6:.1f} MB")

for name, (repo, lic, desc, pats) in SETS.items():
    if a.only and name not in a.only:
        continue
    get(name, repo, lic, desc, pats)
if a.fineweb_files and (not a.only or "fineweb_edu_ja" in a.only):
    files = sorted(f for f in api.list_repo_files(FINEWEB[0], repo_type="dataset") if f.endswith(".parquet"))
    pick = files[:a.fineweb_files]
    get("fineweb_edu_ja", FINEWEB[0], FINEWEB[1], FINEWEB[2], pick + ["README.md"])
