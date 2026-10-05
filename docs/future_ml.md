# Learned role models (optional)

stepscribe's understanding layer is rule-based on purpose: no model, no network, every claim
traceable to measured facts. A learned part classifier can be added as **one more evidence source**
next to the rules, never instead of them.

## The plug-in point (implemented)

A role model is a Python package that registers a callable in the entry-point group
`stepscribe.role_models`:

```toml
[project.entry-points."stepscribe.role_models"]
my_model = "my_package.roles:predict"
```

```python
def predict(part):               # part: stepscribe.models.schema.Part
    return {"motor_mount": 0.7, "arm_link": 0.2}   # role -> probability, roles from roles.yaml
```

For every part, each installed model's probabilities are multiplied by `understanding.ml.weight`
in `stepscribe/knowledge/roles.yaml` (default 0.8) and added to the rule evidence, labelled
`ml:<name>` so the pack shows where a role came from. Nothing is installed or run by default; a
model that fails to load or raises is skipped. See `stepscribe/understanding/ml_roles.py`.

## What a model has to clear before it is worth shipping

- CPU-only inference and a model under 50 MB.
- Role accuracy on `tests/real_labels` that is better than the rules alone:

  ```bash
  python scripts/eval_understanding.py robot_steps tests/real_labels
  ```

  Today the rules score 15 of 15 on the labelled parts (the SO-ARM100 arm and one standoff), so a
  model has to be judged on more and harder labels than that. Contributions of label files for
  other robots are the most useful thing for this.
- No network access at run time.

## Candidate approaches

B-Rep neural networks such as UV-Net and BRepNet, trained on public datasets (ABC, the Fusion 360
Gallery, MFCAD++), could provide a part classifier or face segmenter. None is bundled; training one
needs large CAD datasets and a GPU.
