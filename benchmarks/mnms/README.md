# m&m's (MNMS)

The code uses [`zixianma/mnms`](https://huggingface.co/datasets/zixianma/mnms),
especially the `test_human_verified_filtered` split. Hugging Face data is
loaded directly by the upstream evaluator. Local fixed-subset JSONL files are
not committed.

The pruning implementation is in:

- `code/mtest_mnms/mnms/run_reasoning_tool_rollout.py`;
- `code/mtest_mnms/mnms/summary_kl_core.py`;
- `launchers/analyze_mnms_reasoning_pruning.py`.

Install `code/mtest_mnms/requirements.txt`, then set `PYTHONPATH`:

```bash
pip install -r code/mtest_mnms/requirements.txt
export PYTHONPATH="$PWD/code/mtest_mnms"
```

The retained launchers reproduce the fixed-subset experiments and expect a
JSONL input through their documented local `data/` layout. To use another
download location, create a symlink under `mnms/data/` or edit the launcher
`DATASET` variable locally. Dataset paths are ignored by Git.
