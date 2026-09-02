"""
Builds a calibration dataset from BugsInPy (github.com/soarsmu/BugsInPy).

For each real, verified bug we extract:
  - the BUGGY version of the changed code (pre-fix)
  - the FIXED version of the same code (post-fix)
as paired snippets with ground-truth labels (is_buggy=True / False).

This gives us real, human-verified ground truth to check whether an LLM
critic's stated confidence ("I'm 90% sure this is buggy") actually tracks
reality, instead of trusting the number at face value.

Usage:
    python build_dataset.py --n 40 --out calibration_dataset.json
"""
import argparse
import json
import random
import re
from pathlib import Path

from unidiff import PatchSet

BUGSINPY_ROOT = Path(__file__).parent.parent / "BugsInPy" / "projects"

# Smaller / more tractable projects first -- keeps snippets readable and
# avoids pulling in giant scientific-computing diffs (e.g. pandas, keras)
# for a first pass. Feel free to add more project names from BugsInPy.
DEFAULT_PROJECTS = [
    "PySnooper", "cookiecutter", "httpie", "sanic", "tqdm", "thefuck",
    "tornado", "spacy", "luigi",
]

TRIVIAL_PATTERNS = [
    re.compile(r"^\s*#"),           # comment-only line
    re.compile(r"^\s*(import|from)\s"),  # import changes
    re.compile(r"^\s*\"\"\"|^\s*'''"),   # docstring lines
]


def is_trivial_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(p.match(line) for p in TRIVIAL_PATTERNS)


def extract_hunks(patch_path: Path):
    """Yield (file_path, buggy_code, fixed_code) for each substantive hunk
    in a bug_patch.txt, skipping test files and whitespace/import/comment
    -only changes."""
    try:
        text = patch_path.read_text(errors="ignore")
        patch = PatchSet(text)
    except Exception:
        return

    for patched_file in patch:
        path = patched_file.path
        if not path.endswith(".py"):
            continue
        if "test" in path.lower():
            continue  # we want production code, not the test suite itself

        for hunk in patched_file:
            removed = [l.value.rstrip("\n") for l in hunk.source_lines() if l.is_removed or l.is_context]
            added = [l.value.rstrip("\n") for l in hunk.target_lines() if l.is_added or l.is_context]

            changed_lines = [l.value for l in hunk if (l.is_added or l.is_removed) and not is_trivial_line(l.value)]
            if len(changed_lines) < 1:
                continue  # nothing substantive changed in this hunk
            if len("\n".join(removed)) < 40 or len("\n".join(added)) < 40:
                continue  # too small to be a meaningful snippet

            buggy_code = "\n".join(removed)
            fixed_code = "\n".join(added)
            if buggy_code.strip() == fixed_code.strip():
                continue

            yield path, buggy_code, fixed_code


def build_dataset(projects, n, seed=13):
    random.seed(seed)
    candidates = []

    for project in projects:
        proj_dir = BUGSINPY_ROOT / project / "bugs"
        if not proj_dir.exists():
            continue
        for bug_dir in sorted(proj_dir.iterdir()):
            patch_file = bug_dir / "bug_patch.txt"
            info_file = bug_dir / "bug.info"
            if not patch_file.exists():
                continue

            info = {}
            if info_file.exists():
                for line in info_file.read_text().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        info[k.strip()] = v.strip().strip('"')

            for file_path, buggy_code, fixed_code in extract_hunks(patch_file):
                candidates.append({
                    "project": project,
                    "bug_id": bug_dir.name,
                    "file": file_path,
                    "buggy_commit": info.get("buggy_commit_id", ""),
                    "fixed_commit": info.get("fixed_commit_id", ""),
                    "buggy_code": buggy_code,
                    "fixed_code": fixed_code,
                })

    random.shuffle(candidates)
    selected = candidates[:n]

    # Expand into individual (snippet, label) examples: one bug -> two
    # examples, one buggy and one fixed, so the critic is tested on both
    # "should flag this" and "should NOT flag this" cases.
    examples = []
    for c in selected:
        base_id = f"{c['project']}#{c['bug_id']}"
        examples.append({
            "example_id": f"{base_id}:buggy",
            "project": c["project"],
            "bug_id": c["bug_id"],
            "file": c["file"],
            "code": c["buggy_code"],
            "ground_truth_is_buggy": True,
        })
        examples.append({
            "example_id": f"{base_id}:fixed",
            "project": c["project"],
            "bug_id": c["bug_id"],
            "file": c["file"],
            "code": c["fixed_code"],
            "ground_truth_is_buggy": False,
        })

    random.shuffle(examples)
    return examples, len(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of bugs to sample (each yields 2 examples: buggy+fixed)")
    ap.add_argument("--projects", nargs="*", default=DEFAULT_PROJECTS)
    ap.add_argument("--out", default="calibration_dataset.json")
    args = ap.parse_args()

    examples, total_candidates = build_dataset(args.projects, args.n)

    Path(args.out).write_text(json.dumps(examples, indent=2))
    print(f"Found {total_candidates} candidate bug hunks across {len(args.projects)} projects.")
    print(f"Sampled {args.n} bugs -> {len(examples)} labeled examples (buggy + fixed pairs).")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
