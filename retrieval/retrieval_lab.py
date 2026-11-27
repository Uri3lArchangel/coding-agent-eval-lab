"""Offline whole-file retrieval evaluation. Nothing in the corpus is executed."""
import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

METHODS = ("bm25", "dense", "hybrid")
GROUPS = ("identifier-rich", "paraphrase")
DIM = 96
RRF_K = 60
WORDS = re.compile(r"[A-Za-z][A-Za-z0-9]*")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
STOP = set("a an and as at by for from how in is it of on or that the to was when where which with py ts js go rs".split())


def tokens(text):
    for word in WORDS.findall(text):
        for part in dict.fromkeys([word.lower(), *(p.lower() for p in CAMEL.split(word))]):
            if part not in STOP:
                yield part


def read_fixture(path):
    blob = path.read_bytes()
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    raw = json.loads(blob, object_pairs_hook=no_duplicates)
    def schema(obj, keys):
        return type(obj) is dict and set(obj) == set(keys.split())
    def string(value):
        return type(value) is str and bool(value.strip())
    if (not schema(raw, "name documents queries") or not string(raw["name"])
            or not all(type(raw[key]) is list and raw[key] for key in ("documents", "queries"))):
        raise ValueError("fixture must contain name and nonempty documents/queries lists")
    docs = {}
    for d in raw["documents"]:
        if not schema(d, "path text") or not all(string(d[k]) for k in ("path", "text")):
            raise ValueError("document needs a nonempty path and text")
        name = d["path"]
        if (name in docs or "\\" in name or ":" in name
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError(f"duplicate or invalid document path: {name}")
        docs[name] = d["text"]
    queries = {}
    for q in raw["queries"]:
        if (not schema(q, "id text group relevant") or not all(string(q[k]) for k in ("id", "text"))
                or q["group"] not in GROUPS or type(q["relevant"]) is not list
                or not q["relevant"] or not all(string(p) and p in docs for p in q["relevant"])
                or len(set(q["relevant"])) != len(q["relevant"]) or q["id"] in queries):
            raise ValueError("invalid/duplicate query, group, or relevance label")
        queries[q["id"]] = q
    return raw["name"], docs, [queries[k] for k in sorted(queries)], hashlib.sha256(blob).hexdigest()


def normalize(vec):
    norm = math.sqrt(sum(v*v for v in vec))
    return tuple(v/norm for v in vec) if norm else (0.0,) * len(vec)


def ranking(scores):
    # Do not rank documents with no positive evidence; lexical no-hit results stay empty.
    return sorted((p for p, score in scores.items() if score > 1e-12), key=lambda p: (-round(scores[p], 12), p))


def rrf(*lists):
    score = defaultdict(float)
    for paths in lists:
        for rank, path in enumerate(paths, 1):
            score[path] += 1 / (RRF_K + rank)
    return ranking(score)


class Index:
    def __init__(self, documents, methods):
        self.methods = methods
        self.docs = sorted(documents)
        self.terms = {p: Counter(tokens(p + " " + documents[p])) for p in self.docs}
        df = Counter(w for terms in self.terms.values() for w in terms)
        n = len(self.docs)
        self.idf = {w: math.log1p((n-count+.5)/(count+.5)) for w, count in df.items()}
        if "bm25" in methods or "hybrid" in methods:
            lens = {p: self.terms[p].total() for p in self.docs}
            avg = mean(lens.values()) or 1
            self.postings = defaultdict(list)
            for p, freq in self.terms.items():
                penalty = 1.2*(.25+.75*lens[p]/avg)
                for w, c in freq.items():
                    self.postings[w].append((p, self.idf[w]*2.2*c/(c+penalty)))
        if "dense" in methods or "hybrid" in methods:
            # Random indexing: add signed hashed context coordinates to each token's vector.
            # Each file is one context; distinct terms count once per context while fitting.
            contexts = {w: [0.0]*DIM for w in df}
            for p, terms in self.terms.items():
                digest = hashlib.sha256(p.encode()).digest()
                for j in range(0, 16, 2):
                    k, value = digest[j] % DIM, 1 if digest[j+1] & 1 else -1
                    for w in terms:
                        contexts[w][k] += value
            self.word_vectors = {w: normalize(vec) for w, vec in contexts.items()}
            self.vectors = {p: self.embed(terms) for p, terms in self.terms.items()}
        del self.terms

    def embed(self, terms):
        vec = [0.0]*DIM
        for w in sorted(terms):
            if w in self.word_vectors:
                scale = self.idf[w]*(1+math.log(terms[w]))
                for i, value in enumerate(self.word_vectors[w]):
                    vec[i] += scale * value
        return normalize(vec)

    def search(self, method, query):
        terms = Counter(tokens(query))
        if method != "dense":
            scores = defaultdict(float)
            for w in sorted(terms):
                for path, score in self.postings.get(w, ()):
                    scores[path] += score
            lexical = ranking(scores)
            if method == "bm25":
                return lexical
        vec = self.embed(terms)
        dense = ranking({p: sum(x*y for x,y in zip(vec, rhs)) for p, rhs in self.vectors.items()})
        return dense if method == "dense" else rrf(lexical, dense)


def measures(paths, relevant, k):
    relevant = set(relevant)
    recall = len(set(paths[:k]) & relevant) / len(relevant)
    rr = next((1/i for i, p in enumerate(paths, 1) if p in relevant), 0.0)
    return recall, rr


def evaluate(docs, queries, k, repeats):
    rows, profiles = {}, {}
    indices = {}
    for method in METHODS:
        tracemalloc.start()
        try:
            idx = Index(docs, (method,))
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        indices[method] = idx
        profiles[method] = {"index_retained_python_bytes": retained, "index_peak_python_bytes": peak}
    for q in queries:
        rows[q["id"]] = {"group": q["group"], "relevant": q["relevant"], "scores": {}}
        for method in METHODS:
            ranked = indices[method].search(method, q["text"])
            recall, rr = measures(ranked, q["relevant"], k)
            rows[q["id"]]["scores"][method] = {"recall": recall, "rr": rr, "top_k": ranked[:k]}
    timings = {m: [] for m in METHODS}
    for rep in range(repeats):
        for q in queries:
            # Rotate execution order each repeat without changing rankings or inputs.
            offset = rep % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                start = time.perf_counter_ns()
                indices[method].search(method, q["text"])
                timings[method].append((time.perf_counter_ns()-start)/1_000_000)
    for method, vals in timings.items():
        vals.sort()
        profiles[method].update(warm_p50_ms=median(vals), warm_p95_ms=vals[math.ceil(.95*len(vals))-1], samples=len(vals))
    summary = {}
    for group in ("all", *GROUPS):
        scores = [r["scores"] for r in rows.values() if group == "all" or r["group"] == group]
        if not scores:
            continue
        summary[group] = {"queries": len(scores), "methods": {}}
        for method in METHODS:
            summary[group]["methods"][method] = {
                f"recall@{k}": mean(s[method]["recall"] for s in scores),
                "mrr": mean(s[method]["rr"] for s in scores),
            }
        for base in ("bm25", "dense"):
            deltas = [s["hybrid"]["rr"] - s[base]["rr"] for s in scores]
            summary[group][f"hybrid_vs_{base}_rr"] = {
                "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
                "losses": sum(d < 0 for d in deltas),
            }
    return {"summary": summary, "profile": profiles, "queries": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=Path(__file__).with_name("synthetic.json"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    a = p.parse_args(argv)
    if a.k < 1 or not 1 <= a.repeats <= 10_000:
        p.error("--k must be positive; --repeats must be between 1 and 10000")
    try:
        name, docs, queries, digest = read_fixture(a.fixture)
    except (OSError, ValueError) as e:
        p.error(str(e))
    print(json.dumps({"fixture": name, "fixture_sha256": digest, "files": len(docs),
                      "python": platform.python_version(), "platform": platform.platform(),
                      "dense_dimensions": DIM, "rrf_k": RRF_K,
                      **evaluate(docs, queries, a.k, a.repeats)}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
