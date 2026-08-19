# JigsawBlink

The runner loads the `Jigsaw` configuration and `val` split from
[`BLINK-Benchmark/BLINK`](https://huggingface.co/datasets/BLINK-Benchmark/BLINK).
Raw images and local manifests are not committed.

To create a local manifest:

```bash
cd code/runner/scripts
python materialize_vstar_manifest.py --task jigsaw_blink \
  --out-dir ../../../data/jigsaw_blink
```

JigsawBlink uses its native image-editing/navigation tools, including
`DetectBlackArea`, `AStarWithPixelCoordinate`, `Draw2DPath`, `InsertImage`, and
`Crop`. It must not be evaluated with VStar grounding/OCR tools merely because
both snapshots share some runner infrastructure.

Run the five matched methods with:

```bash
BASE_URL=http://127.0.0.1:8011/v1 \
MODEL=qwen3-vl-8b \
ITEMS_MANIFEST="$PWD/data/jigsaw_blink/jigsaw_blink.jsonl" \
bash run_fiveway.sh
```

The launcher exposes only the Jigsaw-native tool set. Generated outputs belong
under `jigsaw/results/`, which is ignored by Git.
