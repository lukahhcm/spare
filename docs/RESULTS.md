# Results

The paper contains the canonical main table, model breakdowns, and ablations.
This repository keeps generated trajectories and evaluation outputs outside
Git so that the code release stays small and does not expose machine-specific
artifacts.

Across the reported benchmark and backbone settings, SPARE achieves the best
average accuracy among the pruning methods while removing 37.89 to 64.58
percent of reasoning tokens. Use the bundled scorers and manifests to reproduce
the per-benchmark tables under a matched model-serving configuration.
