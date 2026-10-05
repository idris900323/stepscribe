# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""UI worker: runs one analysis in its own process and reports progress through a queue."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any


def run_job(job: dict[str, Any], queue: Any, cmd_q: Any = None, reply_q: Any = None) -> None:
    """Analyse ``job['path']`` and write a pack under ``job['out']``; never raises."""
    from stepscribe.analysis import AnalyzeOptions
    from stepscribe.api import analyze_full
    from stepscribe.pack.design_context import apply_context, load_context
    from stepscribe.pack.writer import write_pack
    from stepscribe.progress import Progress, Tracker

    def push(p: Progress) -> None:
        queue.put(
            {
                "type": "progress",
                "stage": p.stage,
                "message": p.message,
                "fraction": p.fraction,
                "elapsed_s": p.elapsed_s,
                "eta_s": p.eta_s,
            }
        )

    started = time.time()
    try:
        opts = AnalyzeOptions(
            material=job.get("material") or None,
            density=job.get("density"),
            up=job.get("up", "+Z"),
            front=job.get("front", "-Y"),
            check_interference=bool(job.get("interference")),
            no_timestamp=True,
        )
        context_file = job.get("context_file")
        if context_file:
            ctx = load_context(context_file)
            apply_context(opts, ctx)
        tracker = Tracker(push)
        tracker.images_wanted = bool(job.get("images", True))
        analysis = analyze_full(job["path"], opts, tracker)
        if not analysis.parts:
            raise RuntimeError(
                "; ".join(str(e.get("error")) for e in analysis.report.errors)
                or "nothing to analyse"
            )
        folder = write_pack(
            analysis, Path(job["out"]), 60000, context_file, bool(job.get("images", True)), tracker
        )
        tracker.finish("pack written")
        queue.put(
            {
                "type": "done",
                "pack": str(folder),
                "seconds": round(time.time() - started, 1),
                "errors": analysis.report.errors,
            }
        )
        if cmd_q is not None and analysis.report.understanding is not None:
            session_loop(analysis, folder, cmd_q, reply_q)
    except Exception as exc:  # noqa: BLE001
        queue.put(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=6),
            }
        )


IDLE_SECONDS = 3600  # an interview process that nobody talks to for an hour ends


def _state(session: Any, change: Any = None) -> dict[str, Any]:
    from stepscribe.ui.interview_view import question_view, understanding_view

    report = json.loads(session.analysis.report.model_dump_json())
    q = session.next_question()
    return {
        "question": question_view(json.loads(q.model_dump_json())) if q else None,
        "remaining": session.remaining(),
        "asked": session.asked_this_session,
        "can_undo": bool(session.order),
        "change": None
        if change is None
        else {"text": change.text, "seconds": round(change.seconds, 3)},
        "understanding": understanding_view(report),
    }


def session_loop(analysis: Any, folder: Path, cmd_q: Any, reply_q: Any) -> None:
    """Serve interview commands from the page until told to quit (or idle for an hour)."""
    from stepscribe.pack.writer import write_pack
    from stepscribe.render import render_highlight
    from stepscribe.understanding.session import Session

    pipe = analysis.understanding_pipeline
    session = Session(analysis, pipe.opts)
    context_file = folder / "design_context.yaml"
    import multiprocessing as mp

    parent = mp.parent_process()
    idle = 0.0
    while True:
        try:
            cmd = cmd_q.get(timeout=5)
        except Exception:  # noqa: BLE001 - queue.Empty
            idle += 5
            if idle >= IDLE_SECONDS or (parent is not None and not parent.is_alive()):
                return  # nobody is using the session, or the server is gone
            continue
        idle = 0.0
        name = cmd.get("cmd")
        try:
            if name == "quit":
                return
            if name == "state":
                reply_q.put(_state(session))
            elif name == "answer":
                change = session.answer(cmd["id"], cmd["status"], cmd.get("value"), cmd.get("note"))
                reply_q.put(_state(session, change))
            elif name == "undo":
                reply_q.put(_state(session, session.undo()))
            elif name in ("preview", "save"):
                if session.answers:
                    session.save(context_file)
                out = write_pack(
                    analysis,
                    folder.parent,
                    60000,
                    context_file if context_file.is_file() else None,
                    False,
                )
                reply_q.put(
                    {
                        "context_pack": (out / "context_pack.md").read_text(encoding="utf-8"),
                        "path": str(out),
                    }
                )
            elif name == "render":
                png = render_highlight(
                    analysis, list(cmd.get("refs", [])), folder, cmd.get("view", "iso")
                )
                reply_q.put({"path": str(png)})
            else:
                reply_q.put({"error": f"unknown command {name!r}"})
        except Exception as exc:  # noqa: BLE001 - a bad command must not kill the session
            reply_q.put({"error": f"{type(exc).__name__}: {exc}"})


def _dims(p: dict[str, Any]) -> dict[str, Any]:
    """Sizes, volume, surface area and every sized feature of one part (mm, mm2, mm3, g)."""
    m, bb = p["mass"], p["bbox"]["size"]
    hole_rows = []
    for h in p["holes"]:
        depth = "through" if h["is_through"] else (f"{h['depth_mm']:.2f}" if h["depth_mm"] else "?")
        extra = []
        if h.get("counterbore_diameter_mm"):
            extra.append(
                f"counterbore dia {h['counterbore_diameter_mm']:.2f} x {h['counterbore_depth_mm'] or 0:.2f}"
            )
        if h.get("countersink_diameter_mm"):
            extra.append(f"countersink dia {h['countersink_diameter_mm']:.2f}")
        hole_rows.append(
            [
                h["id"],
                f"{h['diameter_mm']:.2f}",
                depth,
                "; ".join(extra) or "-",
                f"{h['edge_distance_mm']:.2f}" if h.get("edge_distance_mm") is not None else "-",
            ]
        )
    return {
        "obb": [f"{v:.2f}" for v in p["obb"]["size_sorted"]],
        "bbox": [f"{bb['x']:.2f}", f"{bb['y']:.2f}", f"{bb['z']:.2f}"],
        "volume_mm3": m["volume_mm3"],
        "area_mm2": m["surface_area_mm2"],
        "mass_g": m["mass_g"],
        "density": m["density_g_cm3"],
        "material": m["material_assumed"],
        "wall": p.get("min_wall_thickness_mm"),
        "holes": hole_rows,
        "cutouts": [
            [
                c["id"],
                c["kind"],
                "; ".join(_level_text(i, lv) for i, lv in enumerate(c["levels"], 1)),
                "through" if c["total_depth_mm"] is None else f"{c['total_depth_mm']:.2f}",
                f"{c['edge_distance_mm']:.2f}" if c.get("edge_distance_mm") is not None else "-",
            ]
            for c in p.get("cutouts", [])
        ],
        "slots": [
            f"{s['id']}: width {s['width_mm']:.2f}, length {s['length_mm']:.2f}, "
            + ("through" if s["is_through"] else f"depth {s['depth_mm'] or 0:.2f}")
            for s in p["slots"]
            if not s.get("cutout_id")
        ],
        "pockets": [
            f"{k['id']}: {k['outline_size'][0]:.2f} x {k['outline_size'][1]:.2f}, depth {k['depth_mm']:.2f}"
            for k in p["pockets"]
            if not k.get("cutout_id")
        ],
        "bosses": [
            f"{b['id']}: dia {b['diameter_mm']:.2f}, height {b['height_mm']:.2f}"
            for b in p["bosses"]
        ],
        "fillets": [
            f"{f['id']}: R{f['radius_mm']:.2f} ({'convex' if f['convex'] else 'concave'})"
            for f in p["fillets"]
        ],
        "chamfers": [
            f"{c['id']}: {c['distance_mm']:.2f} at {c['angle_deg']:.0f} deg" for c in p["chamfers"]
        ],
    }


def _level_text(i: int, lv: dict[str, Any]) -> str:
    """'L1 rectangle 10.00 x 18.00, depth 2.00, centre (50.00, 30.00)' for the Dimensions tab."""
    size = (
        f"dia {lv['width_mm']:.2f}"
        if lv["shape"] == "circle"
        else f"{lv['width_mm']:.2f} x {lv['length_mm']:.2f}"
    )
    corner = f", corner R{lv['corner_radius_mm']:.2f}" if lv.get("corner_radius_mm") else ""
    depth = "through" if lv["depth_mm"] is None else f"depth {lv['depth_mm']:.2f}"
    cu, cv = lv["center_uv"]
    return (
        f"L{i} {lv['shape'].replace('_', ' ')} {size}{corner}, {depth}, centre ({cu:.2f}, {cv:.2f})"
    )


def summarise_pack(folder: Path) -> dict[str, Any]:
    from stepscribe.ui.interview_view import understanding_view

    """Everything the page needs from a finished pack folder."""
    report = json.loads((folder / "data" / "report.json").read_text(encoding="utf-8"))
    asm = report.get("assembly")

    def text(name: str) -> str:
        p = folder / name
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    parts = [
        {
            "id": p["id"],
            "name": p["name"],
            "shape": p["shape_class"]["label"],
            "confidence": p["shape_class"]["confidence"],
            "size": " x ".join(f"{v:.1f}" for v in p["obb"]["size_sorted"]),
            "holes": len(p["holes"]),
            "patterns": [pat["description"] for pat in p["hole_patterns"]],
            "tags": [t["label"] + f" ({t['confidence']:.2f})" for t in p["semantic_tags"]],
            "mass_g": p["mass"]["mass_g"],
            "dims": _dims(p),
            "warnings": p["warnings"],
        }
        for p in report["parts"]
    ]
    images = (
        sorted(
            str(p.relative_to(folder)).replace("\\", "/")
            for p in (folder / "images").rglob("*.png")
        )
        if (folder / "images").is_dir()
        else []
    )
    return {
        "name": folder.name,
        "unit": report["meta"]["original_length_unit"],
        "parts": parts,
        "instances": len(asm["instances"]) if asm else 1,
        "bom": [
            f"{b['part_id']} {b['name']} x {b['quantity']} ({b['category']})" for b in asm["bom"]
        ]
        if asm
        else [],
        "relations": [r["sentence"] for r in asm["relations"]] if asm else [],
        "joints": [
            f"{j['id']}: {j['suggested_fastener'] or 'fastener unknown'}, grip {j['stack_thickness_mm']:.1f} mm"
            for j in asm["fastener_joints"]
        ]
        if asm
        else [],
        "shopping": [f"{s['qty']} x {s['item']}" for s in asm["fastener_shopping_list"]]
        if asm
        else [],
        "total_mass_g": asm["total_mass_g"] if asm else None,
        "warnings": report["warnings"],
        "errors": report["errors"],
        "images": images,
        "context_pack": text("context_pack.md"),
        "understanding": understanding_view(report),
        "overview": text("01_overview.md"),
    }
