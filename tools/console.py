"""会話コンソール: ブラウザでことゆいと話し、返事ごとにグッド・バッドを付ける（持ち主の依頼 2026-11-03）

使い方（リポジトリ直下で）:
    python tools/console.py                     # http://127.0.0.1:8765 をブラウザで開く
    python tools/console.py --light --user=山田  # 共鳴を光の語板で。--user の相手の記憶を使う
    python tools/console.py --ai=qwen3-swallow  # 横に「ローカル AI の返事」も並べる（Ollama、tools/ollama_client.py）
記録: 評点は非公開の置き場の ratings/ratings.jsonl に 1 行ずつ足す（会話の履歴・返事・評点・自信・設定）。
      話し相手の発言は個人の情報なので、リポジトリには入れない。
外の部品は使わない（Python の標準の http.server だけ）。手元の PC の中だけで動かす（127.0.0.1）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kotoyui  # noqa: E402

PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ことゆい 会話コンソール</title><style>
body{font-family:system-ui,"Yu Gothic UI",sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#fcfcfb;color:#0b0b0b}
#log{display:flex;flex-direction:column;gap:10px;margin-bottom:12px}
.u{align-self:flex-end;background:#e9e8e4;padding:8px 12px;border-radius:12px;max-width:80%}
.b{align-self:flex-start;background:#cfe1f7;padding:8px 12px;border-radius:12px;max-width:80%}
.ai{align-self:flex-start;background:#f3f2ee;padding:8px 12px;border-radius:12px;max-width:80%;border:1px dashed #c9c8c2}
.meta{font-size:12px;color:#52514e;margin-top:4px}
button.r{margin-left:6px;border:1px solid #c9c8c2;background:#fff;border-radius:8px;padding:2px 8px;cursor:pointer}
button.r.on{background:#2a78d6;color:#fff;border-color:#2a78d6}
form{display:flex;gap:8px}input{flex:1;padding:10px;font-size:16px;border:1px solid #c9c8c2;border-radius:8px}
#send{padding:10px 16px;border-radius:8px;border:0;background:#2a78d6;color:#fff}
h1{font-size:18px}small{color:#52514e}
</style></head><body><h1>ことゆい 会話コンソール <small id="ver"></small></h1>
<div id="log"></div><form id="f"><input id="t" autocomplete="off" placeholder="話しかける（空で送ると会話を最初から）"><button id="send">送る</button></form>
<script>
let hist=[];const log=document.getElementById('log');
fetch('/info').then(r=>r.json()).then(d=>{document.getElementById('ver').textContent=d.info});
function add(cls,text,meta){const d=document.createElement('div');d.className=cls;d.textContent=text;if(meta){const m=document.createElement('div');m.className='meta';m.append(meta);d.append(m)}log.append(d);window.scrollTo(0,document.body.scrollHeight);return d}
function rate(id,v,btns){fetch('/rate',{method:'POST',body:JSON.stringify({id:id,rating:v})});btns.forEach(b=>b.classList.toggle('on',b.dataset.v==v))}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const t=document.getElementById('t').value.trim();document.getElementById('t').value='';
 if(!t){hist=[];log.innerHTML='';return}
 add('u',t);hist.push(t);
 const r=await (await fetch('/reply',{method:'POST',body:JSON.stringify({history:hist})})).json();
 const meta=document.createElement('span');meta.textContent='自信 '+r.confidence.toFixed(1)+' ';
 const btns=[['good','👍'],['bad','👎']].map(([v,s])=>{const b=document.createElement('button');b.className='r';b.textContent=s;b.dataset.v=v;return b});
 btns.forEach(b=>{b.onclick=()=>rate(r.id,b.dataset.v,btns);meta.append(b)});
 add('b',r.reply,meta);hist.push(r.reply);
 if(r.ai){add('ai','ローカル AI: '+r.ai)}
};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--user", default=None)
    ap.add_argument("--char", default=None)
    ap.add_argument("--ai", default=None, help="Ollama のモデル名。並べて出す（比べる用、評点は付けない）")
    a = ap.parse_args()
    bot = kotoyui.Bot(char=a.char, user=a.user, light=a.light)
    ai = None
    if a.ai:
        from ollama_client import Ollama
        ai = Ollama(a.ai)
    out = os.path.join(kotoyui.PRIVATE, "ratings", "ratings.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lock = threading.Lock()
    turns = {}
    info = f"{kotoyui.__version__}・共鳴 {'光の語板' if a.light else '数え上げ'}・{bot.char.prof.get('苗字', '')}{bot.char.prof.get('名前', '')}"

    def write(rec):
        with lock, open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json; charset=utf-8"):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/info":
                return self.send(200, json.dumps(dict(info=info), ensure_ascii=False))
            self.send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            try:
                d = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self.send(400, json.dumps(dict(error="UTF-8 の JSON を送ってください")))
            if self.path == "/reply":
                hist = [str(t) for t in d.get("history", [])][-20:]
                with lock:
                    reply, conf = bot.reply(hist)
                tid = f"{time.strftime('%Y%m%d%H%M%S')}_{len(turns)}"
                rec = dict(id=tid, time=time.strftime("%Y-%m-%dT%H:%M:%S"), history=hist, reply=reply, confidence=conf,
                           version=kotoyui.__version__, light=a.light, char=bot.char.prof.get("id"), rating=None)
                if ai:
                    try:
                        rec["ai"] = ai.chat(hist)
                    except Exception as e:  # ローカル AI が動いていなくても会話は続ける
                        rec["ai"] = f"（ローカル AI に届かない: {e}）"
                turns[tid] = rec
                return self.send(200, json.dumps(dict(id=tid, reply=reply, confidence=conf, ai=rec.get("ai")), ensure_ascii=False))
            if self.path == "/rate":
                if d.get("id") not in turns or d.get("rating") not in ("good", "bad"):
                    return self.send(400, json.dumps(dict(error="知らない返事か評点です")))
                rec = dict(turns[d["id"]], rating=d["rating"], rated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                write(rec)
                return self.send(200, "{}")
            self.send(404, "{}")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"ことゆい 会話コンソール: http://127.0.0.1:{a.port}（{info}）。評点は {out} に。Ctrl+C で止める", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    if bot.user:
        bot.user.save(len(turns) * 2)


if __name__ == "__main__":
    main()
