# Reading a stepscribe export

## Files

| File | Use it for |
|---|---|
| `MANIFEST.md` | Which file to use where, with sizes and token estimates. Read it first. |
| `<name>_SMALL.md` | Summary, kinematics, top weak spots, open questions, BOM, one paragraph per part. Fits small chats. |
| `<name>_COMPACT.md` | The important parts in full detail, the rest as short paragraphs. |
| `<name>_FULL.md` | Everything: all layers, every part in full, image descriptions, appendix. |
| `chat_bundle/` | SMALL plus a few images, sized for free-tier upload. |
| `pack/` | The multi-file pack (`00_READ_ME_FIRST.md`, `01_overview.md`, `02_assembly.md`, `03_parts/`, `05_understanding.md`). |
| `data/report.json` | Every number, machine readable. `report.schema.json` describes it. |
| `images/` | Labelled renders. The IDs in the labels are the IDs in the text. |

## IDs

PRT part, INS placed part (one copy of a part in the assembly), H hole, P hole pattern, CP cut-out pattern, C contact,
J fastener joint, R relation, S slot, PK pocket, B boss, FL fillet, CH chamfer, L rigid link, KJ kinematic
joint, M mechanism, LP load path, W weak spot, Q question. Face IDs look like F0001.

## Fact or guess

- No prefix: measured from the geometry.
- `Likely:` an inference. The number is a confidence from 0 to 1 and the reason follows.
- `Confirmed by designer:` the designer answered a question. Treat as fact, but if a "conflict" note says it
  disagrees with the geometry, mention the conflict.
- `Designer unsure:` a soft answer. Treat as a guess.

## Units and frames

Millimetres, degrees, grams. The text states the up and front axes. A hole direction points into the material
from the entry face. Part files use the part's own frame; assembly placements give each placed part's position
and rotation in world coordinates.

## What stepscribe cannot know

Threads, material (unless given), tolerances, loads and purpose, unless the designer supplied them. Say so
when a conclusion depends on one of these.
