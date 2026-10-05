"""List every PNG under a folder with its pixel size and file size, as a Markdown table.

Usage:
    python scripts/list_images.py out/gallery            # prints the table
    python scripts/list_images.py out/gallery -o out/IMAGES.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

WHAT = {
    "assembly_iso": "assembly, isometric (front-right-above), part IDs labelled",
    "assembly_front": "assembly, front view with overall width and height dimensions",
    "assembly_top": "assembly, top view with overall width and depth dimensions",
    "assembly_right": "assembly, right view with overall depth and height dimensions",
    "assembly_exploded": "assembly, exploded isometric (hierarchical), part IDs labelled",
}


def describe(name: str) -> str:
    stem = Path(name).stem
    if stem in WHAT:
        return WHAT[stem]
    if stem.endswith("_iso"):
        return f"part {stem[:-4]}, isometric, hole IDs labelled, size caption"
    if stem.endswith("_axis"):
        return f"part {stem[:-5]}, looking down the main hole axis, hole IDs and dimension lines"
    return "image"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    rows = ["| Image | Pixels (w x h) | File size | Shows |", "|---|---|---|---|"]
    for png in sorted(args.folder.rglob("*.png")):
        with Image.open(png) as im:
            w, h = im.size
        rows.append(
            f"| {png.relative_to(args.folder).as_posix()} | {w} x {h} | {png.stat().st_size / 1024:.0f} KB | {describe(png.name)} |"
        )
    text = "\n".join(rows) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"Wrote {args.out} ({len(rows) - 2} images)")
    else:
        print(text)


if __name__ == "__main__":
    main()
