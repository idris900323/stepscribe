---
name: review
description: Analyse a STEP file, export it, and have the mechanical-reviewer agent write an ID-cited design review, then hand over the portable files.
argument-hint: "<file.step>"
disable-model-invocation: true
license: AGPL-3.0-only
---

Review the design in the STEP file `$ARGUMENTS`. If no file was given, ask for the path and stop.

1. Call `start_analysis`, tell the user the ETA, poll `get_job_status` every 20 to 30 seconds with a one-line
   progress message each time, and wait for the job to finish.
2. Call `export_pack` with the job id.
3. Delegate the review to the `mechanical-reviewer` agent. Give it the export folder path and the path of the
   STEP file, and ask it to follow its procedure.
4. Return its review unchanged, then say where `<name>_FULL.md`, `<name>_SMALL.md`, the folder and the zip are,
   and that they can be pasted or uploaded into any other AI tool. If you can send files, send FULL.md and the zip.

Never add measurements the agent or the tools did not produce.
