import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PRIV = _os.environ.get("KOTOYUI_PRIVATE", _os.path.join(_HERE, "..", "..", "..", "Project_Kotoyui_private")).replace("\\", "/").rstrip("/") + "/"   # 非公開の置き場（記録・結果）
import json,sys,io
sys.path.insert(0,_os.path.join(_HERE, ".."))
import memory_board as MB, i35_route_grain as R
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
P=_PRIV
works=json.load(open(P+"i35y/wn50x5_parsed.json",encoding="utf-8")); pb=R.PieceBoards(works)
W=["みさき","歯医者","モモ","誕生日","猫","名前","明日","予定"]
for t in sys.argv[1:]:
    m0=MB.replay(P+f"memory/{t}_memory_wn.json",works,pb.pieces,forget=True)
    d=json.load(open(P+f"memory/{t}_memory_wn.json",encoding="utf-8"))
    a=d["boards"]["日"][0]["keys"]; b=m0.boards["日"][0]["keys"]
    print(t,"日の鍵 再記憶あり",len(a),"なし",len(b))
    print("  ", "; ".join(f"{w}: {a.get('名|'+w,0)} / {b.get('名|'+w,0)}" for w in W))
    up=sorted(((a.get(k,0)-b.get(k,0),k) for k in set(a)|set(b)),reverse=True)[:10]
    print("   上がった鍵",[(k,round(v,1)) for v,k in up])
