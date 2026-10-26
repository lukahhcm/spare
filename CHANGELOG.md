# Changelog

## [2026-10-26]

- Updated the README, method definitions, results, and project page to match the revised paper.
- Distinguished the four-benchmark main evaluation from the three-benchmark aggregate comparison.
- Clarified temporary summary use, rule-based evidence extraction, and sample-weighted metrics.
- Added the arXiv record and complete BibTeX citation, and refreshed the downloadable preprint.

## [2026-08-19 05:55]

- Reorganized the release around one paper-facing framework under
  `benchmarks/`, with shared method and data documentation at the repository
  root.
- Added the latest portable VisualToolBench rollout, SPARE KL policy, pruning
  analysis, paired comparison, and rubric-scoring code without dataset or
  generated-result artifacts.
- Added `run_spare.sh` as the paper-aligned VTB entry point for adaptive
  model-selected summarization and mixed KL/evidence pruning.
- Added `score_all.sh` so the five VTB conditions are scored in one matched
  rubric invocation.
- Replaced the machine-specific Python path in the VTB launchers with the
  configurable `PYTHON_BIN` setting.
- Reframed the main README around textual debt, adaptive summary-conditioned
  replay, KL selection, and visual-evidence preservation.
- Moved detailed result material out of the landing page and kept only the
  paper-level accuracy and pruning summary.
- Passed Python compilation, shell syntax, CLI smoke checks, 42 unit tests,
  Markdown-link checks, and staged-content security/artifact scans.

## [2026-08-19 03:32]

- Created a clean GitHub worktree from the four benchmark code snapshots.
- Excluded datasets, checkpoints, logs, caches, and generated rollout files.
- Updated VStar to the validated Point-fixed runner and added its regression
  tests, five-method launcher, data instructions, and result summary.
- Updated the independent Jigsaw snapshot to the same pruning semantics while
  keeping its native tool profile and separate launcher.
- Added repository-level data, installation, method, and reproducibility docs.
- Removed two upstream machine-specific MNMS import paths and made its
  evaluation scratch directory use the system temporary directory by default.
- Fixed invalid Python escape literals in the MNMS parser and added a parser
  regression test.
- Passed release checks for Python compilation, launch-script syntax, unit
  tests, staged size, credentials, absolute paths, and excluded artifacts.
