"""Trusted, authored repair artifacts. Outcome files cannot supply executable code."""
from itertools import groupby


def previously_seen(values):
    return sorted(set(values))


def stable_unique(values):
    return list(dict.fromkeys(values))


def every_chunk(values, size):
    return [values[index:index + size] for index in range(0, len(values), size)]


def complete_chunks_only(values, size):
    return [values[index:index + size] for index in range(0, len(values) - size + 1, size)]


def off_by_one(low, high):
    return list(range(low, high))


def closed_range(low, high):
    return list(range(low, high + 1))


def all_groups(values):
    return [[key, len(list(group))] for key, group in groupby(values)]


def sorted_groups(values):
    return all_groups(sorted(values))


# Registry is part of this trusted program; there is no dynamic import or eval.
REGISTRY = {
    "stable-unique": {"unordered": previously_seen, "stable": stable_unique},
    "chunk-tail": {"drop-tail": complete_chunks_only, "keep-tail": every_chunk},
    "inclusive-range": {"half-open": off_by_one, "closed": closed_range},
    "run-length": {"sort-first": sorted_groups, "runs": all_groups},
}
