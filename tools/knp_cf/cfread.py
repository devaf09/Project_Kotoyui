"""KNP の格フレーム辞書を Python だけで読む下調べ（KNP 本体は使わない）。"""
import struct, sys, time, re, pickle, os, collections as C
sys.stdout.reconfigure(encoding='utf-8')
PRIV = os.environ.get('KOTOYUI_PRIVATE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'Project_Kotoyui_private'))
E = os.environ.get('KNP_CF_DIR', os.path.join(PRIV, 'data', 'corpus', 'knp_dict', 'dict-bin', 'ebcf')).replace('\\', '/').rstrip('/') + '/'   # KNP の格フレーム辞書（別に入手する）
HERE = os.path.dirname(os.path.abspath(__file__))


def djb(b):
    h = 5381
    for c in b:
        h = (((h << 5) + h) & 0xffffffff) ^ c
    return h


class CDB:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.t = [struct.unpack('<II', self.f.read(8)) for _ in range(256)]

    def get(self, key):
        h = djb(key)
        pos, n = self.t[h & 255]
        if not n:
            return None
        s = (h >> 8) % n
        for k in range(n):
            self.f.seek(pos + 8 * ((s + k) % n))
            hh, p = struct.unpack('<II', self.f.read(8))
            if not p:
                return None
            if hh == h:
                self.f.seek(p)
                kl, vl = struct.unpack('<II', self.f.read(8))
                if self.f.read(kl) == key:
                    return self.f.read(vl)
        return None

    def items(self):
        end = min(x[0] for x in self.t)
        self.f.seek(2048)
        p = 2048
        while p < end:
            kl, vl = struct.unpack('<II', self.f.read(8))
            k = self.f.read(kl)
            v = self.f.read(vl)
            p += 8 + kl + vl
            yield k, v


def main():
    cf = CDB(E + 'cf.db')
    t = time.time()
    keys = [k for k, _ in cf.items()]
    print('cf.db のキー', len(keys), round(time.time() - t, 1), '秒')
    print('キーの例', [k.decode('utf-8', 'replace') for k in keys[:5]])
    for w in ['食べる', '切る', '光る']:
        hit = [k.decode('utf-8', 'replace') for k in keys if k.decode('utf-8', 'replace').startswith(w + '/')]
        print(w, hit[:8])
    m = CDB(E + 'mrph2id.db')
    it = m.items()
    print('mrph2id の例', [(k.decode('utf-8', 'replace'), v.decode('utf-8', 'replace')) for k, v in (next(it) for _ in range(5))])
    k0 = next(k for k in keys if k.decode('utf-8', 'replace').startswith('切る/'))
    print('値の例', cf.get(k0)[:120])


if __name__ == '__main__':
    main()
