#!/usr/bin/env python3
import csv, json, argparse
from pathlib import Path

def phase(root, shards):
    sums={'answer_acc':0.,'inst_align':0.,'tool_acc':0.,'arg_acc':0.,
          'tool_call':0.,'tool_call_error':0.,'summary_calls':0.,
          'external_tool_calls':0.,'external_tool_errors':0.,
          'external_tool_success':0.}
    den={'answer_acc':0,'inst_align':0,'tool_acc':0,'arg_acc':0}
    seen=set()
    for i in range(8):
        ds=json.loads((shards/f'shard_{i:02d}/dataset.json').read_text())
        nt=sum(len(x.get('tools',[])) for x in ds.values()); na=len(ds)
        den['answer_acc']+=na; den['tool_acc']+=nt; den['arg_acc']+=nt; den['inst_align']+=na+nt
        f=next((p for p in (root/f'worker_{i:02d}').glob('*/summary/*.csv')),None)
        if f is None: raise FileNotFoundError(f'no summary for shard {i}')
        rows=list(csv.DictReader(f.open()))
        for r in rows:
            k=r['metric']; v=float(r[list(r)[-1]])
            seen.add(k)
            if k in ('answer_acc','inst_align','tool_acc','arg_acc'): sums[k]+=v/100*({'answer_acc':na,'inst_align':na+nt,'tool_acc':nt,'arg_acc':nt}[k])
            elif k in ('tool_call','tool_call_error','summary_calls',
                       'external_tool_calls','external_tool_errors',
                       'external_tool_success'): sums[k]+=v
    out={k:(sums[k]/den[k]*100 if k in den and den[k] else sums[k]) for k in sums if k in seen}
    out['samples']=den['answer_acc']; out['gt_tool_steps']=den['tool_acc']
    out['avg_summary_calls']=out['summary_calls']/out['samples'] if out['samples'] else 0
    out['avg_external_tool_calls']=out['external_tool_calls']/out['samples'] if out['samples'] else 0
    out['external_tool_success_rate']=(out['external_tool_success']/out['external_tool_calls']*100
                                       if out['external_tool_calls'] else 0)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--shards',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    names=['no_tools','baseline_tools','delete_reasoning','v3']
    results={n:phase(a.root/n,a.shards) for n in names}
    base=results['baseline_tools']
    payload={**results,'delta':{n:{k:results[n][k]-base[k] for k in base if isinstance(base[k],(int,float)) and k not in ('samples','gt_tool_steps')} for n in names if n!='baseline_tools'}}
    a.out.write_text(json.dumps(payload,indent=2)+'\n')
    md=['| experiment | answer_acc | summary_calls/sample | external_tool_calls/sample | external_tool_success_rate |','|---|---:|---:|---:|---:|']
    for n in names:
        x=results[n]; md.append(f"| {n} | {x['answer_acc']:.2f} | {x['avg_summary_calls']:.3f} | {x['avg_external_tool_calls']:.3f} | {x['external_tool_success_rate']:.2f} |")
    a.out.with_suffix('.md').write_text('\n'.join(md)+'\n')
    # Keep one merged, full-dataset result per mode.  Intermediate worker
    # directories are intentionally removed only after this merge succeeds.
    for n in names:
        merged={}
        for i in range(8):
            files=sorted((a.root/n/f'worker_{i:02d}').glob('*/results/*/*.json'))
            if files:
                d=json.loads(files[-1].read_text())
                # Shards reuse local integer item ids; prefix the shard to
                # prevent later shards overwriting earlier failure traces.
                for item_id, detail in d.get('details', {}).items():
                    merged[f'shard_{i:02d}:{item_id}'] = detail
        (a.root/f'{n}_predictions.json').write_text(json.dumps(merged,ensure_ascii=False,indent=2)+'\n')
        import shutil
        for p in (a.root/n).glob('worker_*'):
            if p.is_dir():
                shutil.rmtree(p)
        for p in (a.root/n).glob('config_*.py'):
            p.unlink()
        for p in (a.root/n).glob('worker_*.log'):
            p.unlink()
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
