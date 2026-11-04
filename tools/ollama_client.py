"""ローカル AI（Ollama）につなぐ小さな部品（持ち主の依頼 2026-11-03: 会話の相手・評価役に Qwen3 Swallow 8B などを使う）

Ollama は起動すると http://localhost:11434 で HTTP の API を出す。ここでは /api/chat に JSON を送るだけ（外の部品は使わない）。
    from ollama_client import Ollama
    ai = Ollama("qwen3-swallow")                       # ollama list に出るモデル名
    ai.chat(["こんにちは"])                             # 会話の履歴（交互の発言）→ 次の発言
    ai.rate(["犬の散歩が大変で"], "散歩が大変なんですね。")  # 返事の評点（1〜5）と理由
コマンドから: python tools/ollama_client.py qwen3-swallow "こんにちは"
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class Ollama:
    def __init__(self, model, host=HOST, system="あなたは日本語で自然に雑談する話し相手です。短く 1〜2 文で返事をしてください。", timeout=120, think=False):
        self.model, self.host, self.system, self.timeout, self.think = model, host.rstrip("/"), system, timeout, think   # think=False: 考える途中を省いて速く（思考モデル）

    def _post(self, messages, temperature=0.7, max_tokens=256):
        # max_tokens: 長さの上限。思考モデルは think=False でも考え続けて止まらないことがある（2026-11-04、1 問で 1 万語を超えた）ので必ず切る
        if not self.think and "qwen3" in self.model.lower() and "instruct" not in self.model.lower():   # 考えない版（instruct）はそのままで良い
            return self._raw_nothink(messages, temperature, max_tokens)
        body = dict(model=self.model, messages=messages, stream=False, think=self.think, options=dict(temperature=temperature, num_predict=max_tokens))
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.loads(r.read().decode("utf-8"))["message"]["content"]
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)                # 考える途中の文（思考モデルの <think>）は外す
        if "</think>" in text:                                                    # think=False でも、閉じの印だけ付いて考えが出ることがある
            text = text.rsplit("</think>", 1)[1]
        return text.strip()

    def _raw_nothink(self, messages, temperature, max_tokens):
        """Qwen3 系: 会話の型を自分で組み、考える欄を空にして答えから書かせる（Qwen3 Swallow RL は think=False でも英語で考え始め、
        考える欄を許すと 1,000 語でも終わらなかったため。2026-11-04）。"""
        p = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        body = dict(model=self.model, prompt=p, raw=True, stream=False, options=dict(temperature=temperature, num_predict=max_tokens, stop=["<|im_end|>"]))
        req = urllib.request.Request(self.host + "/api/generate", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))["response"].strip()

    def chat(self, history):
        """会話の履歴（相手の発言から始まる交互の並び）→ 次の発言。"""
        msgs = [dict(role="system", content=self.system)]
        for i, t in enumerate(history):
            msgs.append(dict(role="user" if (len(history) - i) % 2 == 1 else "assistant", content=t))
        return self._post(msgs)

    def rate(self, history, reply):
        """返事の評点（1〜5 の整数）と一言の理由。評価役（持ち主の評点を補うだけ。正解ではない）。"""
        conv = "\n".join(f"{'A' if (len(history) - i) % 2 == 1 else 'B'}: {t}" for i, t in enumerate(history))
        q = (f"次の会話の続きとして、B の返事「{reply}」は自然ですか。1（不自然）〜5（とても自然）の整数で評価し、"
             f"1 行目に数字だけ、2 行目に短い理由を書いてください。\n\n{conv}")
        out = self._post([dict(role="user", content=q)], temperature=0.0, max_tokens=128)
        m = re.search(r"[1-5]", out)
        return (int(m.group()) if m else None), out.split("\n", 1)[-1].strip()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)
    ai = Ollama(sys.argv[1])
    print(ai.chat(sys.argv[2:]))
