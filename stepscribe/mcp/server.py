# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""MCP server (``stepscribe mcp``): follow-up questions about a STEP file from an AI assistant.

Needs the optional extra: ``pip install stepscribe[mcp]``. The analysis tools are read-only; submit_answer only changes the in-memory session.
"""

# No `from __future__ import annotations`: the SDK reads real annotations (Image is a local import).
from typing import TYPE_CHECKING

from stepscribe.headless import set_headless
from stepscribe.mcp import tools

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

INSTRUCTIONS = (
    "Tools for understanding a mechanical design from a STEP file. Pass the absolute path of a "
    ".step/.stp file. Start with get_overview, then get_part, list_holes, get_relations or "
    "find_parts. For a big assembly call start_analysis first and poll get_job_status, then export_pack. To refine the analysis, call get_questions, ask the user, and send each reply "
    "with submit_answer. Statements starting with 'Likely:' are inferences with a confidence. All units "
    "are millimetres. The first call for a file analyses it (can take minutes for big "
    "assemblies); later calls are cached."
)


def build_server() -> "MCPServer":
    """Create the MCP server with every tool registered."""
    try:
        from mcp.server.mcpserver import Image, MCPServer
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit("The MCP server needs the extra: pip install stepscribe[mcp]") from exc

    set_headless()
    server = MCPServer("stepscribe", instructions=INSTRUCTIONS)

    @server.tool()
    def get_overview(path: str) -> str:
        """Overview of the whole design: size, parts, mass, headline features, fasteners."""
        return tools.get_overview(path)

    @server.tool()
    def get_part(path: str, part: str) -> str:
        """Full detail for one part: dimensions, holes, patterns, features. `part` is an ID like PRT003 or a name."""
        return tools.get_part(path, part)

    @server.tool()
    def list_holes(
        path: str, part: str | None = None, min_d: float | None = None, max_d: float | None = None
    ) -> str:
        """Hole table with diameters, depth, edge distance and likely screw standard; optional part and diameter range in mm."""
        return tools.list_holes(path, part, min_d, max_d)

    @server.tool()
    def list_cutouts(path: str, part: str | None = None) -> str:
        """Cut-outs, recesses, stepped openings and edge notches with every level's size, depth and position (mm); optional part."""
        return tools.list_cutouts(path, part)

    @server.tool()
    def get_relations(path: str, part: str | None = None) -> str:
        """Plain-English relations between parts (bolted to, rests on, inserted into, ...)."""
        return tools.get_relations(path, part)

    @server.tool()
    def measure_distance(path: str, a: str, b: str) -> str:
        """Exact minimum distance in mm between two instances (instance ID, name, or a part ID used once)."""
        return tools.measure_distance(path, a, b)

    @server.tool()
    def section(path: str, plane: str, part: str | None = None) -> list[str | Image]:
        """Cut the model with a plane like 'z=12.5': returns a text summary and a drawing (PNG, at most 1200 px)."""
        text, png = tools.section_result(path, plane, part)
        return [text, Image(data=png, format="png")] if png else [text]

    @server.tool()
    def render_view(
        path: str, view: str = "iso", part: str | None = None, highlight: list[str] | None = None
    ) -> list[str | Image]:
        """Labelled render (iso, front, top, right, exploded; PNG, at most 1200 px); optionally one part or highlighted IDs. For a part, the info panel contents come as text."""
        text, png = tools.render_view_result(path, view, part, highlight)
        return [text, Image(data=png, format="png")]

    @server.tool()
    def find_parts(path: str, query: str) -> str:
        """Find parts by name, shape class (plate, bracket, tube...) or semantic tag (motor mount, bearing seat...)."""
        return tools.find_parts(path, query)

    @server.tool()
    def get_questions(path: str) -> list[str | Image]:
        """Open questions about the design (material, purpose, what drives a joint...), most valuable first, with a highlight image for the first one. Ask the user, then call submit_answer."""
        text, png = tools.question_result(path)
        return [text, Image(data=png, format="png")] if png else [text]

    @server.tool()
    def start_analysis(path: str, material: str | None = None) -> str:
        """Start analysing a STEP file in the background; returns at once with a job_id, whether the result is already cached, and an ETA. Use it first for big assemblies (they take minutes), then poll get_job_status."""
        return tools.start_analysis(path, material)

    @server.tool()
    def get_job_status(job_id: str) -> str:
        """Stage, percent and ETA of a job from start_analysis; a short summary once it is done."""
        return tools.get_job_status(job_id)

    @server.tool()
    def export_pack(
        path_or_job_id: str,
        small_budget_tokens: int = 12000,
        compact_budget_tokens: int = 40000,
        chat_max_images: int = 4,
        chat_max_mb: float = 10.0,
        material: str | None = None,
        copy_to: str | None = None,
    ) -> str:
        """Write (or reuse from the cache) the portable export for a STEP path or a finished job_id: returns all file paths, token estimates and the text of the SMALL version. Pass copy_to (a folder, normally the one holding the STEP file) to also save a copy there."""
        return tools.export_pack(
            path_or_job_id,
            small_budget_tokens,
            compact_budget_tokens,
            chat_max_images,
            chat_max_mb,
            material,
            copy_to,
        )

    @server.tool()
    def submit_answer(
        path: str,
        question_id: str,
        status: str = "answered",
        value: str | None = None,
        note: str | None = None,
    ) -> str:
        """Record the user's answer to a question ID from get_questions. status: answered, skipped or not_sure. The analysis updates at once; answers live in memory for this session only."""
        return tools.submit_answer(path, question_id, status, value, note)

    @server.tool()
    def get_shopping_list(path: str) -> str:
        """Fasteners needed, derived from aligned holes across parts."""
        return tools.get_shopping_list(path)

    @server.tool()
    def diff(path_a: str, path_b: str) -> str:
        """What changed between two versions of a design."""
        return tools.diff(path_a, path_b)

    return server


def main() -> None:
    """Run the server over stdio."""
    build_server().run()


if __name__ == "__main__":
    main()
