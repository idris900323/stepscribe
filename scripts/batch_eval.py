"""Run a whole folder of STEP files and report timings, failures and counts (Phase 9).

Usage:
    python scripts/batch_eval.py robot_steps/ --jobs 4 --timeout-s 1800 --csv out/eval.csv

Nothing is written except the optional CSV and an ``--out`` folder for packs (omit it to run the
analysis only). Exit code is 1 if any file failed, which makes it usable as a regression gate.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from stepscribe.analysis import AnalyzeOptions
from stepscribe.batch import Job, options_dict, run_jobs
from stepscribe.io.discovery import find_step_files


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder", type=Path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=None, help="also write packs/reports here")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--limit-mb", type=float, default=None, help="skip files larger than this")
    args = ap.parse_args(argv)

    files = find_step_files(args.folder)
    if args.limit_mb is not None:
        files = [f for f in files if f.stat().st_size <= args.limit_mb * 1e6]
    if not files:
        print("no STEP files found")
        return 1
    out = args.out or Path("out") / "batch_eval"
    opts = options_dict(AnalyzeOptions(no_timestamp=True))
    mode = "pack" if args.out else "analyze"
    jobs = [
        Job(path=str(f), out_dir=str(out), mode=mode, options=opts, images=False) for f in files
    ]
    results = run_jobs(
        jobs,
        args.jobs,
        args.timeout_s,
        on_done=lambda r: print(
            f"{r['status']:6s} {Path(r['file']).name} ({r['seconds']} s)", flush=True
        ),
    )

    failed = [r for r in results if r["status"] != "ok"]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} files ok; total {sum(r['seconds'] for r in results):.0f} s"
    )
    for r in failed:
        print(f"FAILED {Path(r['file']).name}: {'; '.join(str(e) for e in r['errors'])[:200]}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["file", "status", "parts", "holes", "joints", "relations", "warnings", "seconds"]
            )
            for r in results:
                w.writerow(
                    [
                        r["file"],
                        r["status"],
                        r["parts"],
                        r["holes"],
                        r["joints"],
                        r["relations"],
                        r["warnings"],
                        r["seconds"],
                    ]
                )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
