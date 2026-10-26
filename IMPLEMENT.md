# Implementation status

The release contains four benchmark directories matching the paper's main
evaluation: GTA, V*, BLINK-Jigsaw, and m&m's (planning only).
Each adapter retains its benchmark's native tools and evaluation interface.

- `benchmarks/gta/`: GTA / OpenCompass integration and fixed-set verification.
- `benchmarks/vstar/`: visual search, Point/OCR/Crop tools, and five-method launcher.
- `benchmarks/jigsaw/`: BLINK-Jigsaw spatial tools and five-method launcher.
- `benchmarks/mnms/`: m&m's planning-only evaluation and pruning analysis.

The README and project page use V* for the quick-start example. Method definitions,
result tables, dataset preparation, the paper PDF, and the arXiv citation are in
`docs/` and the root README. Datasets, model weights, generated traces, and
credentials are excluded from Git.

See each benchmark's README for setup and regression checks. Reproducing the
paper's results requires the corresponding datasets, model endpoints, and tools.
