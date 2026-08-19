# Implementation status

## Unified paper release

Decision: present SPARE as one method and reproduction framework. Benchmark
directories contain native adapters because their tools and official scorers
differ, but the landing page, method definitions, data instructions, and five
comparison conditions are shared.

Implemented:

- prepared the unified release for `https://github.com/lukahhcm/spare`;
- organized GTA, BLINK-Jigsaw, m&m's, V*, and VisualToolBench under
  `benchmarks/`;
- copied the latest code-only VisualToolBench bundle with multi-round rollout,
  local image tools, summary-conditioned KL, pruning analysis, and rubric
  scoring;
- added a paper-aligned VTB launcher that exposes `summarize_the_task` to the
  agent and applies mixed KL/evidence pruning after the adaptive gate;
- added one matched VTB scoring entry point for all five conditions;
- retained the validated V* Point worker and regression tests;
- aligned the Jigsaw runner with the same pruning semantics while keeping its
  native tools;
- added one paper-facing README plus shared method and data documentation;
- excluded datasets, model weights, generated traces, logs, credentials, and
  machine-specific runtime paths.

Verification completed:

- all published Python files compile;
- all shell launchers pass `bash -n`;
- VTB rollout and scorer CLIs load and expose their argument interfaces;
- 42 unit and regression tests pass, with one endpoint-dependent test skipped;
- tracked content contains no credentials, private absolute paths, datasets,
  generated results, checkpoints, logs, caches, or backup files;
- the staged release contains 201 files and about 2.18 MB of content.

Status: implementation, documentation, project page, and release verification
are complete.
