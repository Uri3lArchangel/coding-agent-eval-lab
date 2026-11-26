"""Evaluate paired, offline coding-agent traces against curated local repairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from solutions import REGISTRY

MAX_BYTES = 1_000_000
MAX_CASES = 128
TASKS_FILE = Path(__file__).with_name("tasks.json")
PROMPTS_FILE = Path(__file__).with_name("prompts.json")
MAX_TRACE = 32
TOOL_EVENTS = {"inspect", "edit", "test"}


class Invalid(ValueError):
    pass


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Invalid(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(token):
    raise Invalid(f"invalid number {token}")


def read(path):
    with path.open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise Invalid(f"{path.name}: exceeds {MAX_BYTES} bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=unique_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as error:
        raise Invalid(f"{path.name}: {error}") from error
    return value, hashlib.sha256(data).hexdigest()


def fields(value, required, context):
    if not isinstance(value, dict) or set(value) != required:
        raise Invalid(f"{context}: expected exactly {', '.join(sorted(required))}")


def int_in(value, lower, upper, context):
    if type(value) is not int or not lower <= value <= upper:
        raise Invalid(f"{context}: expected an integer between {lower} and {upper}")


def suite():
    tasks, digest = read(TASKS_FILE)
    if not isinstance(tasks, dict) or set(tasks) != set(REGISTRY):
        raise Invalid("trusted suite and artifact registry must have identical task IDs")
    for task_id, task in tasks.items():
        fields(task, {"category", "prompt", "cases"}, task_id)
        cases = task["cases"]
        if not isinstance(cases, list) or not 0 < len(cases) <= MAX_CASES:
            raise Invalid(f"{task_id}: invalid cases")
        for case in cases:
            fields(case, {"id", "args", "expected"}, task_id)
    return tasks, digest


def outcomes(path, tasks, prompts):
    document, digest = read(path)
    fields(document, {"schema_version", "label", "source", "outcomes", "prompt_id"}, path.name)
    int_in(document["schema_version"], 1, 1, path.name + " schema_version")
    label, source, items = document["label"], document["source"], document["outcomes"]
    if not isinstance(label, str) or not label.strip() or len(label) > 80:
        raise Invalid(f"{path.name}: invalid label")
    if not isinstance(source, str) or source not in {"scripted", "recorded"}:
        raise Invalid(f"{path.name}: invalid source")
    if not isinstance(items, list) or len(items) != len(tasks):
        raise Invalid(f"{path.name}: expected {len(tasks)} paired outcomes")
    prompt_id = document["prompt_id"]
    if not isinstance(prompt_id, str) or prompt_id not in prompts:
        raise Invalid(f"{path.name}: unknown prompt ID")
    by_id = {}
    for item in items:
        fields(item, {"task_id", "artifact", "trace", "output_tokens"}, path.name)
        task_id, artifact, trace = item["task_id"], item["artifact"], item["trace"]
        if not isinstance(task_id, str) or task_id not in tasks or task_id in by_id:
            raise Invalid(f"{path.name}: unknown or repeated task ID")
        if artifact is not None and (
            not isinstance(artifact, str) or artifact not in REGISTRY[task_id]
        ):
            raise Invalid(f"{path.name}/{task_id}: unknown trusted artifact")
        if not isinstance(trace, list) or len(trace) > MAX_TRACE or any(
            not isinstance(event, str) or event not in TOOL_EVENTS for event in trace
        ):
            raise Invalid(f"{path.name}/{task_id}: invalid trace events")
        int_in(item["output_tokens"], 0, 1_000_000, f"{path.name}/{task_id}: output_tokens")
        by_id[task_id] = item
    return label, source, by_id, digest, prompt_id


def artifact_digest():
    """Fingerprint the shipped trusted repair implementations."""
    return hashlib.sha256(Path(__file__).with_name("solutions.py").read_bytes()).hexdigest()


def equal(left, right):
    """Exact typed comparison of JSON-compatible outputs (True is not 1)."""
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return set(left) == set(right) and all(equal(left[key], right[key]) for key in left)
    return left == right


def grade(task_id, task, result):
    artifact = result["artifact"]
    repair = REGISTRY[task_id].get(artifact)
    failures = []
    if repair is not None:
        for case in task["cases"]:
            try:
                passed = equal(repair(*case["args"]), case["expected"])
            except Exception:
                passed = False
            if not passed:
                failures.append(case["id"])
    return {
        "artifact": artifact,
        "status": "abstained" if repair is None else "failed" if failures else "passed",
        "cases_executed": len(task["cases"]) if repair is not None else 0,
        "failed_cases": failures,
        "reported_tool_events": len(result["trace"]),
        "reported_output_tokens": result["output_tokens"],
    }


def bucket(rows):
    bp = sum(row["baseline"]["status"] == "passed" for row in rows)
    vp = sum(row["variant"]["status"] == "passed" for row in rows)
    regressions = [row["task_id"] for row in rows
                   if row["baseline"]["status"] == "passed" and row["variant"]["status"] != "passed"]
    improvements = [row["task_id"] for row in rows
                    if row["variant"]["status"] == "passed" and row["baseline"]["status"] != "passed"]
    return {"tasks": len(rows), "baseline_passed": bp, "variant_passed": vp,
            "delta_tasks": vp - bp, "regressions": regressions, "improvements": improvements}


def compare(tasks, suite_sha, prompts_sha, baseline, variant, max_regressions, max_slice_regressions):
    rows = []
    for task_id, task in sorted(tasks.items()):
        rows.append({
            "task_id": task_id, "category": task["category"],
            "baseline": grade(task_id, task, baseline[2][task_id]),
            "variant": grade(task_id, task, variant[2][task_id]),
        })
    grouped = {
        category: bucket([row for row in rows if row["category"] == category])
        for category in sorted({row["category"] for row in rows})
    }
    overall = bucket(rows)
    reasons = []
    if len(overall["regressions"]) > max_regressions:
        reasons.append("overall regression budget exceeded")
    for category, report in grouped.items():
        if len(report["regressions"]) > max_slice_regressions:
            reasons.append(f"category regression budget exceeded: {category}")
    return {
        "report_version": 1,
        "suite_sha256": suite_sha,
        "prompt_library_sha256": prompts_sha,
        "artifact_registry_sha256": artifact_digest(),
        "runs": {
            side: {"label": run[0], "source": run[1], "sha256": run[3], "prompt_id": run[4]}
            for side, run in (("baseline", baseline), ("variant", variant))
        },
        "overall": overall, "categories": grouped, "tasks": rows,
        "gate": {"passed": not reasons, "reasons": reasons,
                 "max_regressions": max_regressions,
                 "max_category_regressions": max_slice_regressions},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--max-regressions", type=int, default=0)
    parser.add_argument("--max-category-regressions", type=int, default=0)
    args = parser.parse_args()
    try:
        tasks, digest = suite()
        prompts, prompts_digest = read(PROMPTS_FILE)
        if not isinstance(prompts, dict) or not prompts or any(
            not isinstance(template, str) or template.count("{task}") != 1
            for template in prompts.values()
        ):
            raise Invalid("trusted prompt library must contain templates with one task placeholder")
        for value, name in ((args.max_regressions, "max-regressions"),
                            (args.max_category_regressions, "max-category-regressions")):
            int_in(value, 0, len(tasks), name)
        report = compare(tasks, digest, prompts_digest, outcomes(args.baseline, tasks, prompts),
                         outcomes(args.variant, tasks, prompts),
                         args.max_regressions, args.max_category_regressions)
    except (Invalid, OSError) as error:
        print(f"eval error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
