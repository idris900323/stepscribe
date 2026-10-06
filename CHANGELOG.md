# Changelog

## 0.1.2

- The project is named **stepscribe-cad** everywhere public: the GitHub repository, the README title, the Claude Code marketplace and plugin (commands are `/stepscribe-cad:export`, `/stepscribe-cad:interview`, `/stepscribe-cad:review`). The command line tool is still `stepscribe` and the Python package is still `stepscribe`.
- The summary says "N fastener pieces to buy (screws, nuts, washers)" instead of "N fasteners", which read as a different count than the shopping list.
- `export_pack` with `copy_to` also reports the path of the copied zip.
- The answers file lock handles the Windows permission errors that appear while another writer deletes its lock or reads the file.

## 0.1.1

- README starts with a step-by-step "Start here" (install, the browser page, the Claude Code commands and in what order, where the files end up).
- `/stepscribe:export` saves the export and zip next to the STEP file (new `copy_to` argument of the `export_pack` tool) and reports the full path.
- Image rendering is skipped, instead of crashing, on machines without usable OpenGL.

## 0.1.0

First release.

- **Measuring:** STEP reader (names, colours, tree, units), exact properties, holes (counterbores, countersinks, blind and through, edge distances), patterns, fillets, chamfers, bosses, slots, pockets, cut-outs (through, recess, stepped, notch), wall thickness, shape classes, standard parts (screws, bearings, motors, extrusions, boards).
- **Assembly:** instances and transforms, bill of materials, planar and cylindrical contacts, fastener joints and shopping list, relations as sentences, optional interference check.
- **Understanding layer:** ribs, gussets, symmetry, manufacturing process, links, joints (servo horns included), degrees of freedom, gears, belts, lead screws, load paths, weak spots, stability, part roles, designer questions and answers.
- **Output:** context pack folder, labelled images with dimensions and joint arrows, `report.json` with a JSON Schema.
- **Portable export** (`stepscribe export`): FULL, COMPACT and SMALL Markdown, a chat bundle, a manifest and a zip; reused from disk when nothing changed.
- **AI integrations:** MCP server with background jobs and image content, an Agent Skill, a Claude Code plugin and marketplace with a reviewer agent, a non-interactive `answer` command. None of them start the local web page.
- **Local web page** (`stepscribe ui`) with progress and ETA, an image viewer and the designer interview.
- **Speed:** exact contact detection pruned by face boxes, parallel part analysis, fixes for quadratic loops on parts with hundreds of holes.
- **License:** AGPL-3.0-only with dual licensing, CLA and trademark notice.
