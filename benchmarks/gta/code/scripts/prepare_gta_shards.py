#!/usr/bin/env python3
"""Create deterministic GTA shards for parallel official OpenCompass runs."""
import argparse, json, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shards', type=int, default=8)
    args = ap.parse_args()
    data = json.loads((args.src / 'dataset.json').read_text())
    keys = list(data)
    for si in range(args.shards):
        dst = args.out / f'shard_{si:02d}'
        dst.mkdir(parents=True, exist_ok=True)
        selected = {k: data[k] for j, k in enumerate(keys) if j % args.shards == si}
        (dst / 'dataset.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        shutil.copy2(args.src / 'toolmeta.json', dst / 'toolmeta.json')
        for sample in selected.values():
            for f in sample.get('files', []):
                p, q = args.src / f['path'], dst / f['path']
                q.parent.mkdir(parents=True, exist_ok=True)
                if not q.exists(): shutil.copy2(p, q)
        (dst / 'manifest.json').write_text(json.dumps({'shard': si, 'ids': list(selected), 'count': len(selected)}, indent=2))
    print(json.dumps({'total': len(keys), 'shards': args.shards}, indent=2))

if __name__ == '__main__': main()
