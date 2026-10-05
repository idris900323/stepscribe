"""Score the understanding layer on real robots: precision and recall for joints, mechanisms,
roles and weak spots, and a list of mismatches (this drives tuning of the knowledge YAML files).

Usage:
    python scripts/eval_understanding.py PATH_TO_STEPS tests/real_labels
    python scripts/eval_understanding.py robot_steps tests/real_labels --material alu

PATH_TO_STEPS holds the STEP files (they stay outside the repo); tests/real_labels holds one
hand-written YAML per file (see tests/real_labels/example.yaml for the format).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import analyze_full
from stepscribe.understanding.evaluate import Score, load_labels, score_report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("steps", type=Path, help="folder with the STEP files")
    ap.add_argument("labels", type=Path, help="folder with label YAML files")
    ap.add_argument("--material", default=None)
    args = ap.parse_args()
    files = sorted(p for p in args.labels.glob("*.yaml") if not p.name.startswith("example"))
    if not files:
        print(f"No label files in {args.labels} (copy example.yaml and write your own).")
        return 0
    total: dict[str, Score] = {}
    for lab_path in files:
        labels = load_labels(lab_path)
        step = args.steps / labels["file"]
        if not step.is_file():
            print(f"{lab_path.name}: STEP file {step} not found, skipped")
            continue
        analysis = analyze_full(step, AnalyzeOptions(material=args.material, no_timestamp=True))
        scores = score_report(analysis.report, labels)
        print(f"\n== {labels['file']}")
        for name, sc in scores.items():
            print(
                f"  {name:<11} precision {sc.precision:.2f}  recall {sc.recall:.2f}  ({sc.tp}/{sc.labelled} labelled)"
            )
            for m in sc.misses:
                print(f"      missed: {m}")
            for e in sc.extras[:6]:
                print(f"      extra:  {e}")
            total.setdefault(name, Score()).add(sc)
    print("\n== all files")
    for name, sc in total.items():
        print(
            f"  {name:<11} precision {sc.precision:.2f}  recall {sc.recall:.2f}  ({sc.tp}/{sc.labelled} labelled)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
