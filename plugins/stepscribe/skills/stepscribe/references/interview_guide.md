# Running the designer interview in the chat

The tool asks what geometry cannot tell: material, how parts are made, purpose, payload, what drives a joint.
Each answer changes the analysis (mass, weak spots, tipping, roles), usually in well under a second.

## Offer first

"I have N questions that would sharpen the analysis (material, how it is made, what drives each joint). Want to
go through them? You can skip any."

If the user says no, move on. Do not ask again unless they bring it up.

## One question per message

Use `get_questions` (or `stepscribe questions FILE --json`). Take the first question and write:

1. The question, in plain words.
2. Why it matters (the `why` field), in one short sentence.
3. Options as a numbered list. Mark the tool's guess: `2. 3D printed (FDM) (my guess)`. For a number or free text,
   say what unit or format you want.
4. If the client shows images, include the highlight image for the parts the question is about.
5. A reminder in small words: *reply with a number or text; "skip", "not sure", "back" or "done" also work.*

## Reading the reply

- A number picks that option; text is the answer for free-text questions; "a, c" answers a multi-choice question.
- "skip": submit with status `skipped` (the guess stays, unconfirmed).
- "not sure": submit with status `not_sure` (the guess stays, marked as a soft fact).
- "back": undo the last answer (`stepscribe answer FILE --undo`) and re-ask that question.
- "done": stop asking.

Submit with `submit_answer` (MCP) or `stepscribe answer FILE --id ID --value V [--note ...]`. Question IDs
can change after an answer, so take the next ID from the reply or from a fresh question list.

## After each answer

Read the "what changed" line back in one short sentence ("Mass is now 109 g; tipping angle 3.5 degrees").
If an answer conflicts with the geometry the tool says so; tell the user plainly and keep their answer.

## At the end

Run `export_pack` again so the files include the answers, then say where they are.
