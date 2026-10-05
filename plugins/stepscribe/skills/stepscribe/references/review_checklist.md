# Design review checklist (for you, the AI, while reviewing)

Work from the stepscribe output and tool measurements only. For each finding cite IDs and numbers, and say
whether it rests on measured facts, designer-confirmed facts or "Likely:" inferences. If a needed number is
missing, call a tool (`measure_distance`, `section`, `get_part`, `list_holes`) or say it is unknown.

## Order of work

1. Read MANIFEST, then SMALL or COMPACT or FULL. Look at the assembly images if you can see images.
2. Go through the sections below. Note only what the data supports.
3. Check the weak-spot table (W IDs) first: each has a number, a threshold and a suggestion. Decide which are
   real problems and which are design advice (a "high" row is a rule-of-thumb breach, not a failure).
4. Write the review: Summary, Critical issues, Important, Minor, Questions for the designer, What is done well.

## Fasteners and joints

- Do screw lengths fit the grip (stack thickness)? Short-screw rows (W) and the shopping list tell you.
- Is a load-carrying joint held by a single fastener? Are there fasteners at all where parts only rest?
- Are holes consistent in size across a joint (clearance vs tapped)? Mixed sizes in one stack are suspicious.
- Is there room for a tool and a head (look at the spatial facts and the images)?

## Edge distances and walls

- Hole-to-edge distance against the guideline in the finding (it names the process it assumed). Check the
  assumption: if the process is wrong, the finding changes.
- Thinnest wall against the process minimum. A sampled wall is an estimate; confirm with `section` before
  calling it critical.
- Cut-out webs between openings, and sharp internal corners on printed or bent parts.

## Fits and clearances

- Cylindrical fits: clearance, line-to-line or interference. Is that right for a pivot, a press fit, a bearing seat?
- Overlaps (interference rows): small volumes can be modelling tolerance or a screw inside its hole; large ones
  are real collisions. Report the volume.
- Bearing and shaft seats: do the diameters and widths match a standard size the tool names?

## Kinematics

- Does the joint list match the intended motion? Check axes, parents and children, and `Driven by`.
- Is there a motor or servo for every joint that should be driven? What does the design say about limits?
- Degrees of freedom, closed loops, floating parts (not connected to the ground link).

## Stability and loads

- Centre of mass against the support polygon; the tipping angle and the arm-extended case.
- Load paths: what is the weakest connection on the way to the ground?
- Cantilevers and unsupported spans next to heavy parts (motors, batteries).

## Manufacturability

- Is the inferred process plausible for each part? Is it consistent with the features (wall thickness, sharp
  corners, thread-like holes, draft)?
- Can each part be made as modelled (tool access, minimum radii, overhangs for printing)?

## Assembly order

- Can the parts go together in some order? Look for parts trapped by others, and fasteners that need access
  that a later part blocks.

## Style of the review

- Prioritise by severity. Prefer few well-supported findings over a long list.
- Challenge inferences that look wrong; say why with a number.
- State your assumptions (material, process, loads) when the design context does not give them.
- End with the questions whose answers would change your conclusions.
