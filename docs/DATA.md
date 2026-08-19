# Data preparation

Datasets are not committed to this repository. Download each benchmark from
its official source and place it under the corresponding adapter's ignored
`data/` directory, or pass an explicit path to the launcher.

| Benchmark | Official source | Evaluation split |
| --- | --- | --- |
| VisualToolBench | [xi1ngang/VisualToolBench](https://github.com/xi1ngang/VisualToolBench) | Official evaluation tasks and images |
| m&m's | [zixianma/mnms](https://huggingface.co/datasets/zixianma/mnms) | `test_human_verified_filtered` |
| GTA | [OpenCompass GTA](https://github.com/open-compass/GTA), [Jize1/GTA](https://huggingface.co/datasets/Jize1/GTA) | Official evaluation data |
| V* | [lmms-lab/vstar-bench](https://huggingface.co/datasets/lmms-lab/vstar-bench) | `test` |
| BLINK-Jigsaw | [BLINK-Benchmark/BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | `Jigsaw`, `val` |

## VisualToolBench

Prepare a parquet file with the official task rows and a JSON manifest with the
evaluation IDs. Set `DATA_PATH` and `MANIFEST_PATH` before running the VTB
launchers. Fixed-rubric scoring additionally expects anchor records derived
from the official ground truth and rubric metadata.

## V* and BLINK-Jigsaw

The included materializer downloads the Hugging Face split and creates a local
JSONL manifest with image files:

```bash
cd benchmarks/vstar/code/runner/scripts
python materialize_vstar_manifest.py --task vstar --out-dir ../../../data/vstar

cd ../../../../jigsaw/code/runner/scripts
python materialize_vstar_manifest.py --task jigsaw_blink \
  --out-dir ../../../data/jigsaw_blink
```

## GTA and m&m's

Follow the official repositories for asset download and tool setup. The local
adapters accept explicit dataset and output paths; no machine-specific path is
required by the release configuration.
