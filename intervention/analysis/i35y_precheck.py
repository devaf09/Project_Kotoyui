"""I35y/I35z の回す前の確かめ: 話し相手に 1 回（日本語の本文か）、先生がいれば 1 回（JSON が読めるか）。
使い方: python intervention/analysis/i35y_precheck.py <話し相手> <host> <on|off|unset> [<先生>]   （リポジトリの直下から）"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "intervention")
import chat_partner as C  # noqa: E402

model, host, think = sys.argv[1], sys.argv[2], {"on": True, "off": False, "unset": None}[sys.argv[3]]
teacher = sys.argv[4] if len(sys.argv) > 4 else ""
C.ollama_chat.__defaults__ = C.ollama_chat.__defaults__[:-1] + (think,)
msgs = [dict(role="system", content=C.PARTNER), dict(role="assistant", content="空がきれいだね。"),
        dict(role="user", content="はい、空を飛ぶの？")]
t = C.ollama_chat(host, model, msgs, tries=1)
ok = bool(t) and C.ja(t)
print("precheck partner reply:", repr(t), "japanese:", ok)
if teacher:
    ev, bad = C.teacher_eval(host, teacher, ["相手: 空がきれいだね。"], "空はない。", 1)
    print("precheck teacher:", json.dumps(ev, ensure_ascii=False), "dropped:", bad or "-")
    ok = ok and bad != "読めない"
import os
if os.environ.get("PERSONA"):                                 # I40c: 話し相手がペルソナのことを言うかを 1 回見る（言わなくても止めない）
    import re
    pm = [dict(role="system", content=C.PARTNER + C.PERSONA), dict(role="assistant", content="こんにちは！今日はいい天気だね。"),
          dict(role="user", content="うん。君のこと、何か教えて？")]
    tp = C.ollama_chat(host, model, pm, tries=1)
    hit = re.findall(r"みさき|歯医者|海|誕生日|モモ|猫|辛い", tp or "")
    print("precheck persona reply:", repr(tp), "persona words:", hit or "-")
    ok = ok and bool(tp) and C.ja(tp)
sys.exit(0 if ok else 1)
