# Coding Agent Eval Lab

A credential-free CI gate for paired coding-agent replays. It distinguishes task correctness from what a recorded trace claims, and catches regressions an aggregate score can hide.

```sh
python3 eval_lab.py baseline.json candidate.json   # passes
python3 eval_lab.py baseline.json regression.json  # exits 1: net gain, one regression
python3 -m unittest -v
```

Python 3.10+; no dependencies or network access. The fixtures are authored, synthetic examples for two categories: sequence semantics and boundaries. The task `prompt` fields describe the fixes. `prompts.json` contains minimal and contract-first versioned instructions with a `{task}` placeholder. These are demonstration associations: the filenames do not represent prompts sent to a model, and the pass delta is not measured prompt uplift.

| Candidate vs. baseline | Baseline pass | Variant pass | Paired motion | Default gate |
| --- | ---: | ---: | --- | --- |
| candidate | 2/4 | 4/4 | 2 improvements, 0 regressions | Pass |
| regression | 2/4 | 3/4 | 2 improvements, 1 regression | Fail |

A replay must supply one outcome per known task with a trusted artifact ID (or null for abstention). The evaluator runs the selected, repository-owned repaired function against four deterministic contracts per task. No arbitrary submitted code, dynamic imports, network tools, or shell commands are executed by the evaluator. Traces and token counts are reported metadata, not proof of test execution or model usage. In the demo they are scripted. A passing repair here means only that it passed these curated cases.

The machine-readable JSON shows contract failures, absolute counts, paired improvement and regression IDs, and category-specific gates. It fingerprints the suite, prompt library, repair registry, and both run files. Exit codes are 0 (gate passed), 1 (gate failed, report emitted), and 2 (invalid input, no report). Both regression budgets default to zero; optional flags can relax them separately. The zero-regression gate is not an absolute quality bar: a uniformly poor unchanged run may pass it, so inspect counts too.

The tasks and implementations are bundled and visible, not a secret holdout, and are too small for statistical or model-quality claims. They illustrate replayable evaluator engineering, not a benchmark against SWE-bench. The original [SWE-bench harness documentation](https://www.swebench.com/SWE-bench/api/harness/) is useful background on separating repairs from regressions in code evaluation.

## Retrieval evaluation

The [offline retrieval experiment](retrieval/) compares identifier-aware BM25, a local dense vector baseline, and reciprocal-rank fusion over the same files. It reports ranked results, relevance, latency including query encoding, and Python allocation measurements. Its original synthetic fixture checks retrieval behavior separately from the repair replay. It does not measure a pretrained encoder or a coding agent.
