"""持ち主の評点（ratings.jsonl の 👍・👎）が付いた返事を、別のローカル AI でも評点し直し、合い方（AUC）を比べる（2026-11-04）
実行（Project_Kotoyui の直下で）: python experiments/rerate_owner.py --ai=<モデル名>
"""
import argparse
import json
import os
import sys

KOTO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.environ.get("KOTOYUI_PRIVATE", os.path.join(KOTO_ROOT, "..", "Project_Kotoyui_private"))   # 非公開の置き場（資料・結果）
sys.path.insert(0, os.path.join(KOTO_ROOT, "tools"))
from ollama_client import Ollama  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ai", required=True)
a = ap.parse_args()
rs = [json.loads(l) for l in open(os.path.join(PRIV, "ratings", "ratings.jsonl"), encoding="utf-8")]
rs = [r for r in rs if r.get("rating") in ("good", "bad")]
ai = Ollama(a.ai, timeout=300)
out = []
for r in rs:
    sc, why = ai.rate(r["history"], r["reply"])
    out.append(dict(rating=r["rating"], old=r.get("ai_score"), new=sc))
good = [x["new"] for x in out if x["rating"] == "good" and x["new"]]
bad = [x["new"] for x in out if x["rating"] == "bad" and x["new"]]
auc = sum(1.0 if g > b else 0.5 if g == b else 0.0 for g in good for b in bad) / max(1, len(good) * len(bad))
rec = dict(ai=a.ai, n=len(out), good_mean=sum(good) / len(good), bad_mean=sum(bad) / len(bad), auc=auc)
print(rec, flush=True)
with open(os.path.join(PRIV, "results", "rerate_owner.txt"), "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
