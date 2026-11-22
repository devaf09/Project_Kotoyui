"""OpenAI Chat Completions 互換の口を呼ぶ小さな道具（先生役をクラウドの口でも呼べるようにする。I48bn・chat_partner の先生）。

- 口の URL（.../v1 まで）・モデル名・鍵の環境変数名は呼ぶ側が渡す。鍵は環境変数から読むだけで、どこにも書かない。
- 呼んだ回数を数え、上限（Budget）を越えたら BudgetOver を投げて止める（無償の枠などを使い切らないため）。
- 429・5xx は少し待って数回だけ再試行する。
- 返り値: dict(text=本文, finish=finish_reason, reasoning=考えた文（あれば）, model=返ってきたモデル名, usage=トークンの数)。
  本文が空で finish が length の時は、考えたトークンで max_tokens を使い切った（呼ぶ側で「読めない」とは別に数える）。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


class BudgetOver(RuntimeError):
    pass


class Budget:
    """1 回の実行で口を呼ぶ回数の上限。"""

    def __init__(self, limit):
        self.limit, self.used = int(limit), 0

    def take(self):
        if self.used >= self.limit:
            raise BudgetOver(f"呼ぶ回数の上限 {self.limit} に達した")
        self.used += 1


def chat(url, model, messages, key_env, budget, temperature=0.2, max_tokens=2048, json_mode=True,
         reasoning_effort=None, chat_template_kwargs=None, timeout=330, retries=4, wait=60, extra=None):
    key = os.environ.get(key_env, "")
    if not key:
        raise RuntimeError(f"環境変数 {key_env} が空（鍵が入っていない）")
    body = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=False)
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    if extra:                                             # 口が受ける他の欄（例: seed）
        body.update(extra)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = None
    for k in range(retries + 1):
        budget.take()
        req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=data, method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
            ch = (out.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            return dict(text=(msg.get("content") or "").strip(), finish=ch.get("finish_reason"),
                        reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
                        model=out.get("model", model), usage=out.get("usage", {}))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 429 or e.code >= 500:
                time.sleep(wait * (k + 1))
                continue
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                detail = ""
            return dict(text="", finish=f"error {e.code}", reasoning="", model=model, usage={}, error=detail)
        except (urllib.error.URLError, TimeoutError) as e:
            last = f"{type(e).__name__}"
            time.sleep(wait * (k + 1))
    return dict(text="", finish=f"error {last}", reasoning="", model=model, usage={})
