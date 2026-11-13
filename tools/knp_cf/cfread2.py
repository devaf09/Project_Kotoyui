"""格フレーム: 3 動詞の上位 10 と、語板の動詞すべてを読む時間・覆う割合を測る。辞書の中身は保存しない。"""
import json, os, sys, time, collections as C
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfread import CDB, E, PRIV

BOARDS = os.path.join(PRIV, 'data', 'boards', 'v2', 'content_boards.json')


def surface(tok, id2):
    """'123+45?67' の形を語に。? は同じ語の書き換えなので最初を取る。"""
    out = []
    for part in tok.split('+'):
        w = id2.get(part.split('?')[0])
        if w is None:
            return None
        out.append(w.split('/')[0])
    return ''.join(out)


def parse(raw):
    f = raw.rstrip(b'\0').split(b'\0')
    head, feat = f[0].decode('utf-8', 'replace'), f[3].decode('utf-8', 'replace')
    cases = []
    for i in range(4, len(f) - 2, 3):
        cases.append((f[i].decode('utf-8', 'replace'), f[i + 1].decode('ascii', 'replace')))
    return head, feat, cases


def frames(cf, dat, key):
    v = cf.get(key)
    if not v:
        return
    for a in v.decode('ascii').split('/'):
        addr, size = a.split(':')
        dat.seek(int(addr))
        yield parse(dat.read(int(size)))


def main():
    t0 = time.time()
    id2 = {}
    lemmas = C.Counter()
    for k, v in CDB(E + 'mrph2id.db').items():
        w = k.decode('utf-8', 'replace')
        id2[v.decode('ascii', 'replace')] = w
        lemmas[w.split('/')[0]] += 1
    print('語の ID', len(id2), round(time.time() - t0, 1), '秒', flush=True)

    cf = CDB(E + 'cf.db')
    dat = open(E + 'cf.dat', 'rb')
    keys = [k for k, _ in cf.items()]
    plain = C.defaultdict(list)
    for k in keys:
        s = k.decode('utf-8', 'replace')
        if '+' not in s and '~' not in s:
            plain[s.split('/')[0]].append(k)

    feats = C.Counter()
    for w, key in [('食べる', '食べる/たべる'), ('切る', '切る/きる'), ('光る', '光る/ひかる')]:
        n = 0
        cnt = C.defaultdict(C.Counter)
        for head, feat, cases in frames(cf, dat, key.encode('utf-8')):
            n += 1
            for x in feat.split():
                feats[x.split(':')[0]] += 1
            if '和フレーム' in feat:      # 全部を足した枠があれば二重に数えない
                continue
            for name, ex in cases:
                for it in ex.split():
                    if ':' not in it:
                        continue
                    tok, c = it.rsplit(':', 1)
                    s = surface(tok, id2)
                    if s:
                        cnt[name][s] += float(c)
        print('==', key, '枠', n, flush=True)
        for name in sorted(cnt, key=lambda x: -sum(cnt[x].values()))[:6]:
            print(' ', name, '延べ', int(sum(cnt[name].values())), '種類', len(cnt[name]), [(a, int(b)) for a, b in cnt[name].most_common(10)])
    print('素性の例', feats.most_common(15))

    B = json.load(open(BOARDS, encoding='utf-8'))
    for pos in ['動詞', '形容詞', '名詞']:
        L = [b['name'] for b in B if b['pos'] == pos]
        print(pos, len(L), '格フレームの見出しがある', sum(w in plain for w in L), 'ID に語がある', sum(w in lemmas for w in L))
    for w in ['跨がる', '跨る', '逃げ帰る', '竹', 'おじいさん', '玉手箱', '小槌', '羽衣', '反物', '桃太郎', '浦島']:
        print(w, '見出し' if w in plain else '-', 'ID' if w in lemmas else '-')

    # 語板の動詞すべてを読む時間（中身は持たない）
    t = time.time()
    frames_n = 0
    bytes_n = 0
    pairs = C.Counter()
    verbs = [b['name'] for b in B if b['pos'] == '動詞']
    done = 0
    for w in verbs:
        for key in plain.get(w, []):
            v = cf.get(key)
            for a in v.decode('ascii').split('/'):
                addr, size = a.split(':')
                dat.seek(int(addr))
                raw = dat.read(int(size))
                bytes_n += len(raw)
                frames_n += 1
                _, _, cases = parse(raw)
                for name, ex in cases:
                    if name in ('ガ', 'ヲ', 'ニ'):
                        pairs[name] += ex.count(':')
        done += 1
        if done % 500 == 0:
            print(' 読んだ動詞', done, round(time.time() - t), '秒', flush=True)
    print('語板の動詞を全部読む:', round(time.time() - t, 1), '秒', '枠', frames_n, '読んだ', round(bytes_n / 1e9, 2), 'GB', '格ごとの名詞の組', dict(pairs))


if __name__ == '__main__':
    main()
