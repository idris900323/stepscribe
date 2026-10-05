---
name: export
description: Analyse a STEP file and hand over the portable Markdown export (FULL, COMPACT, SMALL, chat bundle, zip) with a guide to which file to use where.
argument-hint: "<file.step>"
disable-model-invocation: true
license: AGPL-3.0-only
---

Export the STEP file `$ARGUMENTS` with stepscribe. If no file was given, ask for the path and stop.

Follow the `stepscribe` skill. Specifically:

1. Call the MCP tool `start_analysis` with the absolute path. Tell the user the ETA it returns and whether the
   result is already cached.
2. Poll `get_job_status` every 20 to 30 seconds. After each poll write one short progress line (stage and
   percent). Do not describe results before the job reports it is done. If it reports an error, show the error
   and stop.
3. Call `export_pack` with the job id and `copy_to` set to the folder that holds the STEP file, so the result
   lands next to it. Read `MANIFEST.md`, then the SMALL text it returns.
4. Reply with: what the design is in two to four sentences with IDs (size, parts, how it moves, top risks), then
   where the files are (the copy next to the STEP file, as a full path the user can open) and the "which file should I use?" guide from the manifest: FULL for big-context chats,
   COMPACT for normal chats, `chat_bundle/` for free tiers and small models, the zip for sharing a person.
5. If you can send files, send `<name>_FULL.md` and the zip.

Every number you state comes from the tool output. Never invent a measurement.
