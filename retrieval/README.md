# Retrieval evaluation lab

This offline experiment compares whole-file BM25, a corpus-fitted dense encoder with cosine similarity, and reciprocal rank fusion (RRF). All three rank the same candidate files without executing code or using query labels to fit the index.

```sh
python3 retrieval/retrieval_lab.py --repeats 80 > report.json
python3 -m unittest discover -s retrieval -v
```

Python 3.10+; standard library only.

## Protocol

- **BM25:** entire file plus path, retaining intact identifiers and splitting camel case and underscores. An inverted index scores positive matches only.
- **Dense:** stable signed context sketches derived from the same files, normalized term vectors, and IDF-weighted log-frequency sums produce 96-dimensional vectors. This is a local random-indexing baseline, not a pretrained encoder or approximate index.
- **Hybrid:** reciprocal-rank fusion over the complete positive-evidence lists, with a constant of 60 and stable path tie-breaking.
- **Report:** macro Recall@k, untruncated MRR, per-query results, paired wins/ties/losses, and a fixture fingerprint. Warm latency includes query encoding, scoring and ranking; index allocation measurements use tracemalloc, not process RSS.

## Measured result

One CPython 3.12.14/Linux x86-64 run on 24 original synthetic files and 16 queries with 1,280 timed searches per method:

| Method | Recall@5 | MRR | Warm p95 |
| --- | ---: | ---: | ---: |
| BM25 | 0.7500 | 0.6808 | 0.022 ms |
| Dense | 0.8125 | 0.6896 | 0.199 ms |
| Hybrid | 0.7500 | 0.6860 | 0.217 ms |

Hybrid won one reciprocal-rank comparison versus BM25 and lost one versus dense, both on paraphrases. The fixture is synthetic and small; these results do not measure real-repository retrieval or downstream patch success. Timing varies by machine.

Pass `--fixture` with JSON matching the bundled schema. Freeze candidate scope, source commits, and relevance labels before external evaluation. No external corpus is bundled. See [Agent Retrieval Bench](https://github.com/eyuansu62/agent-retrieval-bench) and its [corpus license notes](https://github.com/eyuansu62/agent-retrieval-bench/blob/main/DATA_LICENSE.md), and [Embeddings vs. Grep](https://arxiv.org/abs/2611.20455) for motivation.
