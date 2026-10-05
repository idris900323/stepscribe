# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Portable export: one folder and one zip that work in any chat or tool.

``export()`` writes ``<name>_stepscribe/`` with a FULL, a COMPACT and a SMALL Markdown file, a small
upload set for chats (``chat_bundle/``), the multi-file pack, images and the data files. The result
is cached on disk by the file's hash, the analysis options and the design context, so exporting
an unchanged file again does no geometry work.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from stepscribe import __version__, config
from stepscribe.analysis import AnalyzeOptions
from stepscribe.api import Analysis, analyze_full
from stepscribe.describe.budget import estimate_tokens
from stepscribe.exporter_text import clean, compose_full, compose_small
from stepscribe.models.schema import Report, json_schema
from stepscribe.pack.design_context import apply_context, load_context
from stepscribe.pack.writer import _safe, write_pack_build
from stepscribe.progress import Tracker

STAMP = ".stepscribe_export.json"
IMAGE_PRIORITY = ("assembly_iso", "assembly_exploded", "kinematics_iso", "assembly_front")
MIN_PANEL_SCALE = 0.55  # part-image info panels use a 20 px font; keep at least 11 px
DEFAULT_SMALL_TOKENS = 12000
DEFAULT_COMPACT_TOKENS = config.DEFAULT_BUDGET_TOKENS


class ExportFile(BaseModel):
    """One file of the export."""

    path: str
    bytes: int
    tokens: int | None = None
    purpose: str


class ExportResult(BaseModel):
    """Where everything went, how big it is, and what was left out."""

    name: str
    folder: str
    full: str
    compact: str
    small: str
    manifest: str
    manifest_json: str
    zip: str | None = None
    chat_bundle: list[str] = Field(default_factory=list)
    dropped_images: list[str] = Field(default_factory=list)
    files: list[ExportFile] = Field(default_factory=list)
    tokens: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False

    def small_text(self) -> str:
        """Contents of ``<name>_SMALL.md``."""
        return Path(self.small).read_text(encoding="utf-8")


# ------------------------------------------------------------------ cache key
def export_key(
    path: Path,
    options: AnalyzeOptions,
    context_file: str | Path | None,
    params: dict[str, Any],
) -> str:
    """Digest of everything that changes the export."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(json.dumps(asdict(options), sort_keys=True, default=str).encode())
    if context_file:
        h.update(Path(context_file).read_bytes())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(__version__.encode())
    return h.hexdigest()


def _read_stamp(folder: Path) -> dict[str, Any] | None:
    try:
        return json.loads((folder / STAMP).read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (OSError, ValueError):
        return None


def cached_export(
    path: str | Path,
    out_dir: str | Path,
    options: AnalyzeOptions | None = None,
    context_file: str | Path | None = None,
    **params: Any,
) -> ExportResult | None:
    """The finished export for these inputs if one is on disk, else None (no geometry work)."""
    p = Path(path)
    opts = _with_context(options or AnalyzeOptions(), context_file)
    folder = Path(out_dir) / f"{_safe(p.stem)}_stepscribe"
    stamp = _read_stamp(folder)
    if not stamp or stamp.get("key") != export_key(p, opts, context_file, _params(params)):
        return None
    result = ExportResult(**stamp["result"])
    result.cached = True
    return result


def load_result(folder: str | Path) -> ExportResult | None:
    """The result recorded in an export folder, if it is one."""
    stamp = _read_stamp(Path(folder))
    return ExportResult(**stamp["result"]) if stamp else None


def _params(params: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "small_budget_tokens",
        "compact_budget_tokens",
        "chat_max_images",
        "chat_max_mb",
        "embed_images",
        "make_zip",
        "images",
    )
    defaults: dict[str, Any] = {
        "small_budget_tokens": DEFAULT_SMALL_TOKENS,
        "compact_budget_tokens": DEFAULT_COMPACT_TOKENS,
        "chat_max_images": 4,
        "chat_max_mb": 10.0,
        "embed_images": False,
        "make_zip": True,
        "images": True,
    }
    return {k: params.get(k, defaults[k]) for k in keys}


def _with_context(options: AnalyzeOptions, context_file: str | Path | None) -> AnalyzeOptions:
    opts = AnalyzeOptions(**asdict(options))
    if context_file:
        apply_context(opts, load_context(context_file))
    return opts


# ------------------------------------------------------------------ the export
def export(
    path: str | Path,
    out_dir: str | Path = "stepscribe_export",
    options: AnalyzeOptions | None = None,
    context_file: str | Path | None = None,
    small_budget_tokens: int = DEFAULT_SMALL_TOKENS,
    compact_budget_tokens: int = DEFAULT_COMPACT_TOKENS,
    chat_max_images: int = 4,
    chat_max_mb: float = 10.0,
    embed_images: bool = False,
    make_zip: bool = True,
    images: bool = True,
    tracker: Tracker | None = None,
    analysis: Analysis | None = None,
) -> ExportResult:
    """Analyse *path* (unless the export is already on disk) and write the export folder."""
    p = Path(path)
    params = _params(
        {
            "small_budget_tokens": small_budget_tokens,
            "compact_budget_tokens": compact_budget_tokens,
            "chat_max_images": chat_max_images,
            "chat_max_mb": chat_max_mb,
            "embed_images": embed_images,
            "make_zip": make_zip,
            "images": images,
        }
    )
    opts = _with_context(options or AnalyzeOptions(), context_file)
    root = Path(out_dir)
    name = _safe(p.stem)
    folder = root / f"{name}_stepscribe"
    key = export_key(p, opts, context_file, params)
    stamp = _read_stamp(folder)
    if stamp and stamp.get("key") == key:
        result = ExportResult(**stamp["result"])
        result.cached = True
        return result
    if folder.exists() and any(folder.iterdir()) and stamp is None:
        raise ValueError(f"{folder} exists and is not a stepscribe export; choose another --out")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    tracker = tracker or Tracker()
    tracker.images_wanted = images
    if analysis is None:
        analysis = analyze_full(p, opts, tracker)
    report = analysis.report
    if not analysis.parts:
        errors = "; ".join(str(e.get("error")) for e in report.errors)
        raise ValueError(f"nothing could be analysed: {errors or 'no parts found'}")
    tracker.stage("export", "writing the export")
    build = write_pack_build(
        analysis,
        root,
        compact_budget_tokens,
        context_file,
        images,
        tracker,
        folder=folder / "pack",
    )
    pack = build.folder
    warnings: list[str] = []
    no_ts = bool(report.meta.options.get("no_timestamp"))

    def put(rel: str, text: str) -> Path:
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean(text), encoding="utf-8", newline="\n")
        return target

    img_rels: list[str] = []
    if (pack / "images").is_dir():
        shutil.copytree(pack / "images", folder / "images")
        img_rels = _ordered_images(folder)
    full_text = compose_full(build, report, name, img_rels, no_ts)
    put(f"{name}_FULL.md", full_text)
    if embed_images and img_rels:
        put(f"{name}_FULL_embedded.md", _embed(full_text, folder))
    compact = (pack / "context_pack.md").read_text(encoding="utf-8")
    compact_note = (
        f"> Compact version: budget {compact_budget_tokens} tokens, about "
        f"{estimate_tokens(compact)} used. Parts that did not fit have a short paragraph here and "
        f"full text in `{name}_FULL.md`.\n\n"
    )
    put(f"{name}_COMPACT.md", compact_note + compact)
    small = compose_small(report, build.sections, name, build.up, build.front, small_budget_tokens)
    put(f"{name}_SMALL.md", small)

    (folder / "data").mkdir(exist_ok=True)
    shutil.copyfile(pack / "data" / "report.json", folder / "data" / "report.json")
    put("data/report.schema.json", json.dumps(json_schema(), indent=2, sort_keys=True))
    if context_file and Path(context_file).is_file():
        shutil.copyfile(context_file, folder / "data" / "design_context.yaml")

    bundle, dropped = _chat_bundle(folder, name, img_rels, report, chat_max_images, chat_max_mb)
    files = _list_files(folder, name, bundle, dropped)
    manifest_md = _manifest_md(name, files, dropped, small_budget_tokens, compact_budget_tokens)
    put("MANIFEST.md", manifest_md)
    put(
        "manifest.json",
        json.dumps(
            {"name": name, "files": [f.model_dump() for f in files], "dropped_images": dropped},
            indent=2,
        ),
    )
    zip_path = _zip(folder) if make_zip else None
    tokens = {f.path: f.tokens for f in files if f.tokens is not None and f.path.endswith(".md")}
    result = ExportResult(
        name=name,
        folder=str(folder),
        full=str(folder / f"{name}_FULL.md"),
        compact=str(folder / f"{name}_COMPACT.md"),
        small=str(folder / f"{name}_SMALL.md"),
        manifest=str(folder / "MANIFEST.md"),
        manifest_json=str(folder / "manifest.json"),
        zip=str(zip_path) if zip_path else None,
        chat_bundle=[str(folder / "chat_bundle" / b) for b in bundle],
        dropped_images=dropped,
        files=files,
        tokens=tokens,
        warnings=warnings + list(report.warnings),
    )
    (folder / STAMP).write_text(
        json.dumps({"key": key, "result": result.model_dump()}, indent=1), encoding="utf-8"
    )
    # the zip is built from the folder, so the stamp is left out of it (see _zip)
    tracker.finish("export written")
    return result


# ------------------------------------------------------------------ pieces
_IMG_LINK = re.compile(r"\]\((images/[^)]+\.png)\)")


def _embed(text: str, folder: Path) -> str:
    def repl(m: re.Match[str]) -> str:
        data = (folder / m.group(1)).read_bytes()
        return f"](data:image/png;base64,{base64.b64encode(data).decode('ascii')})"

    return _IMG_LINK.sub(repl, text)


def _ordered_images(folder: Path) -> list[str]:
    """Every image under ``images/``: assembly views, kinematics, then parts."""
    rels = [p.relative_to(folder).as_posix() for p in (folder / "images").rglob("*.png")]

    def rank(rel: str) -> tuple[int, str]:
        name = Path(rel).name
        return (0 if name.startswith("assembly_") else 1 if "kinematics" in name else 2, rel)

    return sorted(rels, key=rank)


def _priority_images(img_rels: list[str], report: Report) -> list[str]:
    """Images in the order a chat should receive them."""
    by_stem = {Path(r).stem: r for r in img_rels}
    out = [by_stem[s] for s in IMAGE_PRIORITY if s in by_stem]
    u = report.understanding
    if u and u.weak_spots:
        counts: dict[str, int] = {}
        for w in u.weak_spots:
            for ref in w.refs:
                if ref.startswith("PRT"):
                    counts[ref] = counts.get(ref, 0) + 1
        for pid, _n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            for kind in ("axis", "iso"):
                rel = by_stem.get(f"{pid}_{kind}")
                if rel:
                    out.append(rel)
                    break
            if len(out) > len(IMAGE_PRIORITY):
                break
    for r in img_rels:
        if r not in out:
            out.append(r)
    return out


def _shrink(src: Path, dst: Path, max_bytes: int, part_image: bool) -> bool:
    """Write *src* to *dst* within *max_bytes*, downscaling as far as legibility allows."""
    from PIL import Image

    if src.stat().st_size <= max_bytes:
        shutil.copyfile(src, dst)
        return True
    floor = MIN_PANEL_SCALE if part_image else 0.25
    img = Image.open(src)
    scale = 0.9
    while scale >= floor:
        small = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS
        )
        small.save(dst, optimize=True)
        if dst.stat().st_size <= max_bytes:
            return True
        scale -= 0.1
    dst.unlink(missing_ok=True)
    return False


def _chat_bundle(
    folder: Path,
    name: str,
    img_rels: list[str],
    report: Report,
    max_images: int,
    max_mb: float,
) -> tuple[list[str], list[str]]:
    """Copy SMALL.md and the best few images under the size cap; list what was dropped."""
    bundle = folder / "chat_bundle"
    bundle.mkdir(exist_ok=True)
    small = folder / f"{name}_SMALL.md"
    shutil.copyfile(small, bundle / small.name)
    left = int(max_mb * 1_000_000) - small.stat().st_size
    names = [small.name]
    dropped: list[str] = []
    for n, rel in enumerate(_priority_images(img_rels, report)):
        if n >= max_images:
            dropped.append(f"{rel} (over the {max_images}-image limit)")
            continue
        src = folder / rel
        dst = bundle / Path(rel).name
        if left > 0 and _shrink(src, dst, left, "/parts/" in rel or "PRT" in Path(rel).name):
            left -= dst.stat().st_size
            names.append(dst.name)
        else:
            dropped.append(f"{rel} (does not fit the {max_mb:g} MB cap)")
    return names, dropped


_PURPOSE = {
    "_FULL.md": "Everything, no size limit: all layers and every part in full detail.",
    "_FULL_embedded.md": "Same as FULL with the images embedded as base64 (large).",
    "_COMPACT.md": "Budgeted version with the most important parts in full detail.",
    "_SMALL.md": "Short version for free-tier chats and small local models.",
    "MANIFEST.md": "This list.",
    "manifest.json": "This list, machine readable.",
    "report.json": "The complete measured report (all numbers).",
    "report.schema.json": "JSON Schema of report.json.",
    "design_context.yaml": "The designer's intent and answers.",
}


def _purpose(rel: str) -> str:
    for suffix, text in _PURPOSE.items():
        if rel.endswith(suffix):
            return text
    if rel.startswith("chat_bundle/"):
        return "Upload set for a chat: send every file in this folder together."
    if rel.startswith("images/"):
        return "Rendered view with labels that use the IDs in the text."
    if rel.startswith("pack/"):
        return "The multi-file context pack (read 00_READ_ME_FIRST.md first)."
    return "Supporting file."


def _list_files(folder: Path, name: str, bundle: list[str], dropped: list[str]) -> list[ExportFile]:
    del name, bundle, dropped
    rows: list[ExportFile] = []
    for f in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != STAMP):
        rel = f.relative_to(folder).as_posix()
        tokens = None
        if f.suffix == ".md":
            tokens = estimate_tokens(f.read_text(encoding="utf-8"))
        rows.append(
            ExportFile(path=rel, bytes=f.stat().st_size, tokens=tokens, purpose=_purpose(rel))
        )
    return rows


def _manifest_md(
    name: str, files: list[ExportFile], dropped: list[str], small: int, compact: int
) -> str:
    top = [f for f in files if "/" not in f.path]
    lines = [
        f"# {name}: export manifest",
        "",
        "## Which file should I use?",
        "",
        f"- Big-context chat or document: `{name}_FULL.md`",
        f"- Normal chat (plus a few images): `{name}_COMPACT.md` (budget {compact} tokens)",
        f"- Free tier or small/local model: upload everything in `chat_bundle/`, or paste "
        f"`{name}_SMALL.md` (budget {small} tokens)",
        f"- Sharing with a person: the zip `{name}_stepscribe.zip`",
        "- Your own tools: `data/report.json` and `data/report.schema.json`",
        "",
        "## Files",
        "",
        "| File | Size (bytes) | Tokens (estimate) | What it is |",
        "|---|---|---|---|",
    ]
    for f in top + [
        f for f in files if "/" in f.path and not f.path.startswith(("pack/", "images/"))
    ]:
        tok = "" if f.tokens is None else str(f.tokens)
        lines.append(f"| `{f.path}` | {f.bytes} | {tok} | {f.purpose} |")
    n_pack = sum(f.path.startswith("pack/") for f in files)
    n_img = sum(f.path.startswith("images/") for f in files)
    lines += [
        "",
        f"`pack/` holds {n_pack} files (the multi-file context pack); `images/` holds {n_img} images.",
    ]
    if dropped:
        lines += ["", "## Left out of the chat bundle", "", *[f"- {d}" for d in dropped]]
    lines += ["", "Token estimates are characters divided by 4, rounded up."]
    return "\n".join(lines)


def _zip(folder: Path) -> Path:
    """``<folder>.zip`` with the folder's contents under the folder name, in sorted order."""
    target = folder.parent / f"{folder.name}.zip"
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(p for p in folder.rglob("*") if p.is_file() and p.name != STAMP):
            info = zipfile.ZipInfo(f"{folder.name}/{f.relative_to(folder).as_posix()}")
            info.date_time = (2020, 1, 1, 0, 0, 0)  # fixed, so the zip is reproducible
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, f.read_bytes())
    return target


def write_index(results: list[ExportResult], out_dir: Path) -> Path:
    """``INDEX.md`` and ``all_exports.zip`` for a folder run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Exports", "", "| Name | FULL | COMPACT | SMALL | Zip |", "|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: r.name.lower()):
        base = Path(r.folder).name
        zip_name = Path(r.zip).name if r.zip else "-"
        lines.append(
            f"| {r.name} | {base}/{r.name}_FULL.md | {base}/{r.name}_COMPACT.md | "
            f"{base}/{r.name}_SMALL.md | {zip_name} |"
        )
    index = out_dir / "INDEX.md"
    index.write_text(clean("\n".join(lines)), encoding="utf-8", newline="\n")
    target = out_dir / "all_exports.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in sorted(results, key=lambda r: r.name.lower()):
            if r.zip and Path(r.zip).is_file():
                zf.write(r.zip, Path(r.zip).name)
        zf.write(index, "INDEX.md")
    return index
