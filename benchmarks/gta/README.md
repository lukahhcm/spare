# GTA

Download the official benchmark from
[`open-compass/GTA`](https://github.com/open-compass/GTA) or
[`Jize1/GTA`](https://huggingface.co/datasets/Jize1/GTA). Dataset files and
images are not included here.

The key integration points are:

- `code/reasoning_prune/`: Summary-KL scoring and structured-evidence policy;
- `code/opencompass/icl_agent_inferencer.py`: native OpenCompass agent hook;
- `code/scripts/prepare_gta_local_dataset.py`: filter to locally available
  tools;
- `code/scripts/prepare_gta_shards.py`: deterministic sharding;
- `code/scripts/run_gta_8gpu.py`: multi-endpoint launcher.

See `code/inference_README.md` and `code/method.md` for the adapter contract and
five-method definitions. Put downloaded data under `gta/data/` or pass an
explicit path to the preparation scripts.
