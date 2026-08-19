#!/usr/bin/env python3
"""Prepare a GTA manifest containing only tools available in the live server."""
import argparse, json, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--dst', type=Path, required=True)
    ap.add_argument('--tools', nargs='+', required=True)
    args = ap.parse_args()
    src = json.loads((args.src / 'dataset.json').read_text())
    allowed = set(args.tools)
    out = {}
    excluded = {}
    for k, sample in src.items():
        names = {t['name'] for t in sample.get('tools', [])}
        if names <= allowed:
            out[k] = sample
        else:
            excluded[k] = sorted(names - allowed)
    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / 'image').mkdir(exist_ok=True)
    for k, sample in out.items():
        for f in sample.get('files', []):
            p = args.src / f['path']
            q = args.dst / f['path']
            q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists(): shutil.copy2(p, q)
    (args.dst / 'dataset.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
    shutil.copy2(args.src / 'toolmeta.json', args.dst / 'toolmeta.json')
    (args.dst / 'filter_manifest.json').write_text(json.dumps({
        'source': str(args.src), 'allowed_tools': sorted(allowed),
        'included_ids': list(out), 'excluded': excluded,
        'included_count': len(out), 'excluded_count': len(excluded)},
        ensure_ascii=False, indent=2))
    print(json.dumps({'included': len(out), 'excluded': len(excluded)}, indent=2))

if __name__ == '__main__': main()
