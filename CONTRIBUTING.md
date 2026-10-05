# Contributing to stepscribe

## License and CLA

stepscribe is licensed under the [GNU AGPL-3.0](LICENSE) and is also offered under a commercial license (see [COMMERCIAL.md](COMMERCIAL.md)). To make that possible, every contributor signs the [CLA](CLA.md), which lets the maintainer relicense contributions. When you open your first pull request, a bot asks you to sign by commenting on the PR. You only sign once.

Every Python file under `stepscribe/` starts with the SPDX line `# SPDX-License-Identifier: AGPL-3.0-only` and a copyright line. Keep them on new files.

## Development

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,mcp]"
pytest
ruff check .
mypy
```

The agent skill lives in `skills/stepscribe/`. After changing it run `python scripts/sync_skills.py` to update the copies in `plugins/` and `.github/skills/` and the skill zip (CI fails if they differ). `python scripts/check_release.py` checks that the package, plugin and marketplace share one version.

Rules the code follows: no network or LLM calls in the core, deterministic output, every number sourced from the STEP file, tests for new behaviour, full typing (mypy strict). Third-party STEP files stay out of the repository (`robot_steps/` is gitignored).

Files use LF line endings (`.gitattributes` enforces this).
