"""List the real methods of an OCP class (Rule 1: never invent OCP API names).

Usage:
    python scripts/introspect.py OCP.BRepGProp BRepGProp
    python scripts/introspect.py OCP.BRepGProp          # list module members
"""

from __future__ import annotations

import importlib
import sys


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    module = importlib.import_module(argv[0])
    target = getattr(module, argv[1]) if len(argv) > 1 else module
    for name in sorted(n for n in dir(target) if not n.startswith("__")):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
