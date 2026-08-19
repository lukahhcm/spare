#!/usr/bin/env python3
"""Run four GTA ablations on eight model endpoints."""
import argparse, os, subprocess, time
from pathlib import Path

TEMPLATE = '''from lagent.agents import ReAct
from opencompass.models import OpenAI
from opencompass.models.lagent import LagentAgent
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
reader=dict(input_columns=['dialogs','resources'],output_column='gt_answer',train_split='test',test_split='test')
infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template='{{questions}}'),retriever=dict(type=ZeroRetriever),inferencer=dict(type=AgentInferencer,infer_mode='every'))
eval_cfg=dict(evaluator=dict(type=GTABenchEvaluator,mode='every'))
datasets=[dict(abbr='gta_local',type=GTABenchDataset,path={dataset!r},reader_cfg=reader,infer_cfg=infer_cfg,eval_cfg=eval_cfg)]
models=[dict(abbr={abbr!r},type=LagentAgent,agent_type=ReAct,prune_mode={mode!r},reasoning_output=True,v5_prune_strategy={v5_prune_strategy!r},v5_threshold={v5_threshold!r},v5_min_high_tokens={v5_min_high_tokens!r},v5_summary_policy={v5_summary_policy!r},max_turn=10,tool_server={tool_server!r},llm=dict(type=OpenAI,path={model_name!r},key='EMPTY',openai_api_base={url!r},query_per_second=100,max_seq_len=32768,retry=2,temperature=0.0),batch_size=1)]
infer=dict(partitioner=dict(type=SizePartitioner,max_task_size=1000,gen_task_coef=1),runner=dict(type=LocalRunner,task=dict(type=OpenICLInferTask),max_num_workers=1))
'''


def render_config(**kwargs):
    values = {
        "v5_prune_strategy": "visual",
        "v5_threshold": 0.2,
        "v5_min_high_tokens": 2,
        "v5_summary_policy": "autonomous",
        **kwargs,
    }
    values["dataset"] = str(values["dataset"])
    return TEMPLATE.format(**values)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--opencompass',type=Path,required=True); ap.add_argument('--shards',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--python',required=True); ap.add_argument('--start-port',type=int,default=8000); ap.add_argument('--endpoint-port',type=int,help='Send every shard to one shared OpenAI-compatible endpoint.'); ap.add_argument('--tool-server',default='http://127.0.0.1:16182'); ap.add_argument('--model-name',default='qwen3-vl-8b'); ap.add_argument('--only',choices=['no_tools','baseline_tools','delete_reasoning','visual_evidence_only','v3','v5']); ap.add_argument('--label',help='Output/config label for one parameterized condition.'); ap.add_argument('--v5-prune-strategy',choices=['none','selective_delete','visual'],default='visual'); ap.add_argument('--v5-threshold',type=float,default=0.2); ap.add_argument('--v5-min-high-tokens',type=int,default=2); ap.add_argument('--v5-summary-policy',choices=['autonomous','every','final'],default='autonomous'); ap.add_argument('--shard-start',type=int,default=0, help='First shard index to run.'); ap.add_argument('--shard-count',type=int,default=8, help='Number of consecutive shards to run.')
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); inference_root=Path(__file__).resolve().parents[1]; env['PYTHONPATH']=str(inference_root)+':'+str(a.opencompass.parent/'agentlego')+':'+str(a.opencompass); env['NO_PROXY']=env['no_proxy']='127.0.0.1,localhost'
    modes = ((a.only,) if a.only else ('no_tools','baseline_tools','delete_reasoning','v3'))
    mode_cfg = {'no_tools':'no_tools','baseline_tools':None,'delete_reasoning':'delete_reasoning','visual_evidence_only':'visual_evidence_only','v3':'v3','v5':'v5'}
    for mode in modes:
        label = a.label or mode
        phase=a.out/label; phase.mkdir(exist_ok=True); procs=[]
        shard_indices = range(a.shard_start, a.shard_start + a.shard_count)
        for i in shard_indices:
            ds=a.shards/f'shard_{i:02d}'; cfg=phase/f'config_{i:02d}.py'
            # Keep task IDs descriptive for cross-model comparisons; inference
            # settings remain identical across models.
            model_tag = Path(a.model_name).name.lower().replace('/', '_')
            abbr=f'{model_tag}-{label}-s{i:02d}'
            port = a.endpoint_port if a.endpoint_port is not None else a.start_port + i
            cfg.write_text(render_config(dataset=ds,abbr=abbr,mode=mode_cfg[mode],model_name=a.model_name,url=f'http://127.0.0.1:{port}/v1/chat/completions',tool_server=a.tool_server,v5_prune_strategy=a.v5_prune_strategy,v5_threshold=a.v5_threshold,v5_min_high_tokens=a.v5_min_high_tokens,v5_summary_policy=a.v5_summary_policy))
            wd=phase/f'worker_{i:02d}'; wd.mkdir(exist_ok=True)
            log=(phase/f'worker_{i:02d}.log').open('w')
            cmd=[a.python,str(a.opencompass/'run.py'),str(cfg),'--work-dir',str(wd),'--max-num-workers','1']
            procs.append((i,subprocess.Popen(cmd,cwd=str(a.opencompass),env=env,stdout=log,stderr=subprocess.STDOUT),log))
        print(f'{mode}: started {a.shard_count} workers',flush=True)
        rc=[]
        for i,p,log in procs:
            code=p.wait(); log.close(); rc.append(code); print(f'{mode}: shard {i} exit {code}',flush=True)
        if any(rc): raise SystemExit(f'{mode} failed: {rc}')
        print(f'{mode}: all {a.shard_count} workers complete',flush=True)

if __name__=='__main__': main()
