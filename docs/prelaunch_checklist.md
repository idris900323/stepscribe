# Release checklist

What has been checked, and what still needs a person or a first run on GitHub.

## Checked

- [x] Lint, types and tests pass (ruff, mypy, pytest).
- [x] `claude plugin validate plugins/stepscribe --strict` and `claude plugin validate . --strict` pass.
- [x] A release dry run builds the wheel, the sdist and `stepscribe-skill.zip`; `scripts/check_release.py --dist dist` finds the license in all three and one version everywhere.
- [x] The local web page was exercised in a real browser: upload, progress and ETA, every tab, Copy, the image viewer, the designer interview and the zip download.
- [x] A real 15-part assembly (SO-ARM100 5-DOF) packs in about 3 minutes with images and the interference check.
- [x] The MCP server, `export`, `questions`, `answer` and the job runner never import `stepscribe.ui` or `webbrowser` (tested in a scripted session).
- [x] Role labels for the SO-ARM100 and one standoff are in `tests/real_labels/`; the rules match all of them.

## Open

- [ ] CI green on GitHub for all OS and Python combinations (`.github/workflows/ci.yml` has not run yet). Golden files hold floats; if a platform differs, use tolerance-aware comparisons.
- [ ] Accuracy table: compare 10 hole diameters, 2 wall thicknesses and 3 overall sizes on real parts with a CAD system and record the results in the README.
- [ ] Spot-check the knowledge YAML entries against datasheets (NEMA 17/23 spacing and pilot, 608/627/683 bearings, Raspberry Pi mounting pattern, 2020/2040 profiles, GT2 pitch, T8 lead screws) and resolve each `TODO: verify`.
- [ ] More real label files in `tests/real_labels/` for other robots; tune `stepscribe/knowledge/roles.yaml` where roles are wrong.
- [ ] Run the LLM adapter once against a real local model (for example through Ollama).
- [ ] Walk through `docs/plugin_testing.md` once in a real Claude Code session.
- [ ] Legal review of the CLA and the commercial terms before accepting outside contributions or selling licenses.
- [ ] Hidden-line (HLR) drawings are not implemented.
- [ ] Very large assemblies (dozens of heavy parts) can take many minutes; contact detection is the cost.
