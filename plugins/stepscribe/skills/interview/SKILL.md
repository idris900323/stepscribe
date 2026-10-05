---
name: interview
description: Analyse a STEP file, then ask the designer the questions the geometry cannot answer (material, process, purpose, what drives each joint), one at a time in the chat, and re-export with the answers.
argument-hint: "<file.step>"
disable-model-invocation: true
license: AGPL-3.0-only
---

Run the designer interview for the STEP file `$ARGUMENTS`. If no file was given, ask for the path and stop.

1. Call `start_analysis` (it may already be cached), tell the user the ETA, and poll `get_job_status` every 20 to
   30 seconds with a one-line progress message each time. Wait for the job to finish.
2. Call `get_questions`. Say: "I have N questions that would sharpen the analysis. Want to go through them? You
   can skip any." If the user declines, call `export_pack` and stop.
3. Follow `references/interview_guide.md` in the `stepscribe` skill: one question per message, the reason in one
   short sentence, numbered options with the tool's guess marked, the highlight image when you can show images,
   and a reminder that "skip", "not sure", "back" and "done" work.
4. Submit each reply with `submit_answer`. Read back the "what changed" line in one sentence.
5. When the user says done (or there are no more questions), call `export_pack` again so the files include the
   answers. Hand over the files as in the `stepscribe` skill, step 8.

Everything stays in this chat. Never invent a number.
