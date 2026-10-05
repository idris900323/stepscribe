---
name: stepscribe
description: Analyze STEP CAD files (.step/.stp) of mechanical designs and robots - dimensions, holes, fasteners, assembly structure, how the design moves, weak spots - and hand over portable Markdown. Use when the user shares or mentions a STEP file, asks what a CAD model is or how it works, wants a design review, or wants a text description of a 3D model.
license: AGPL-3.0-only
compatibility: Needs Python 3.11+ with the stepscribe package (pip install stepscribe, or "stepscribe[mcp]" for the MCP server). Works offline.
---

# stepscribe: understand a STEP file

stepscribe reads a STEP file and writes measured facts (sizes, holes, fasteners, contacts) plus rule-based
inferences ("Likely: ..." with a confidence). It uses no AI model and no network. You add the judgement.

**Never invent a number.** Every dimension, count or position you state comes from stepscribe output or a
tool call. If a number is missing, measure it with a tool or say it is unknown.

## 1. Check the environment

Run `python scripts/check_env.py` (in this skill's folder). It prints the stepscribe version and whether
the MCP server dependencies are present. It never installs anything.

If stepscribe is missing, **ask the user before installing** (`pip install stepscribe`, or
`pip install "stepscribe[mcp]"` for the MCP server). If installing is not possible here, say so plainly and
suggest running `stepscribe export robot.step` on their own machine and uploading the exported files
(`robot_FULL.md`, or the `chat_bundle/` folder for small chats) instead.

## 2. Prefer the MCP tools, otherwise the CLI

MCP tools: `start_analysis`, `get_job_status`, `export_pack`, `get_overview`, `get_part`, `list_holes`,
`list_cutouts`, `get_relations`, `measure_distance`, `section`, `render_view`, `find_parts`,
`get_questions`, `submit_answer`, `get_shopping_list`, `diff`.

CLI equivalents: `stepscribe export FILE`, `stepscribe questions FILE --json`, `stepscribe answer FILE --id ID
--value V` (also `--skip`, `--not-sure`, `--undo`), `stepscribe kinematics FILE`, `stepscribe weak-spots FILE`.

## 3. Expect long runs

Real assemblies take minutes. Call `start_analysis` (it returns at once with an ETA and whether the result is
cached), tell the user the ETA ("about 3 minutes"), then poll `get_job_status` every 20 to 30 seconds and give
a short progress line each time. Do not claim results before the job is done.

## 4. Export, then read

Call `export_pack` with the job id and `copy_to` set to the folder of the STEP file, so the result sits next to it. Read `MANIFEST.md`, then `<name>_SMALL.md` (or COMPACT or FULL if your
context allows). If you can see images, look at the assembly iso and exploded views. The layout of the files
is in `references/pack_format.md`.

## 5. Give a short summary

Two to five sentences with IDs: what it is, overall size, number of parts, how it moves (joints and what
drives them), and the top risks. Separate measured facts, designer-confirmed facts and "Likely:" inferences.

## 6. Offer the interview

Say: "I have N questions that would sharpen the analysis. Want to go through them? You can skip any." Then
follow `references/interview_guide.md`: one question per message, numbered options with the tool's guess
marked, and skip / not sure / back / done understood from the reply. Submit each answer with `submit_answer`
(or `stepscribe answer`). Export again at the end.

Everything stays in this chat: ask questions as messages, show images inline as image content (or describe
them and give the file path), and hand results over as files and text.

## 7. Review when asked

Follow `references/review_checklist.md`. Prioritise by severity. Challenge inferences that look wrong. State
your assumptions (material, process, loads) when the design context does not give them.

## 8. Always finish with the portable files

In one or two lines say where `<name>_FULL.md`, `<name>_SMALL.md`, the folder and the zip are, and that they
can be pasted or uploaded into any other AI tool or document. If you can send files, send FULL.md and the zip.
