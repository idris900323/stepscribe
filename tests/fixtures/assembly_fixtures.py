"""Assembly fixtures written with XCAF so that prototypes are genuinely shared (ground truth known)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from build123d import Box, Cylinder, Location, Pos, Rot, Shape
from generate_fixtures import FIXTURE_DIR
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool


@dataclass
class Node:
    """An instance in the assembly tree: a prototype part, or a subassembly with children."""

    name: str
    proto: str | None = None
    loc: Location = field(default_factory=Location)
    children: list[Node] = field(default_factory=list)


def _set_name(label: TDF_Label, name: str) -> None:
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))


def write_assembly(path: Path, protos: dict[str, Shape], root: Node) -> Path:
    """Write a STEP assembly: *protos* are shared prototypes, *root* the instance tree."""
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.InitDocument(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    proto_labels: dict[str, TDF_Label] = {}
    for name, shape in protos.items():
        label = tool.AddShape(shape.wrapped, False)
        _set_name(label, name)
        proto_labels[name] = label

    def build(node: Node) -> TDF_Label:
        asm = tool.NewShape()
        _set_name(asm, node.name)
        for child in node.children:
            target = proto_labels[child.proto] if child.proto else build(child)
            comp = tool.AddComponent(asm, target, child.loc.wrapped)
            _set_name(comp, child.name)
        return asm

    build(root)
    tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer.Write(str(path))
    return path


def _plate(with_holes: bool = True) -> Shape:
    p = Box(60, 60, 5)
    if with_holes:
        for x in (-20, 20):
            for y in (-20, 20):
                p = p - Pos(x, y, 0) * Cylinder(1.7, 6)
    return p


def two_plates_assembly(out: Path = FIXTURE_DIR) -> Path:
    """One 'Plate' prototype used twice: BasePlate, and TopPlate resting on it (grip 10, 4 joints)."""
    root = Node(
        "TwoPlates",
        children=[
            Node("BasePlate", "Plate", Location()),
            Node("TopPlate", "Plate", Location((0, 0, 5))),
        ],
    )
    return write_assembly(out / "two_plates_assembly.step", {"Plate": _plate()}, root)


def tapped_assembly(out: Path = FIXTURE_DIR) -> Path:
    """Clearance plate (Ø3.4 x4, 5 thick) on a block with four blind Ø2.5 x 8 tapped holes."""
    block = Box(60, 60, 12)
    for x in (-20, 20):
        for y in (-20, 20):
            block = block - Pos(x, y, 6 - 4) * Cylinder(1.25, 8)
    root = Node(
        "Tapped",
        children=[
            Node("Block", "Block", Pos(0, 0, -6)),
            Node("Cover", "Plate", Pos(0, 0, 2.5)),
        ],
    )
    return write_assembly(out / "tapped_assembly.step", {"Block": block, "Plate": _plate()}, root)


def shaft_in_bearing_assembly(out: Path = FIXTURE_DIR) -> Path:
    """Ø8 shaft in a Ø8.06 bore (diametral clearance 0.06)."""
    block = Box(30, 30, 20) - Cylinder(4.03, 22)
    shaft = Cylinder(4.0, 40)
    root = Node(
        "ShaftAssy",
        children=[Node("Housing", "Housing", Location()), Node("Shaft", "Shaft", Location())],
    )
    return write_assembly(
        out / "shaft_in_bearing_assembly.step", {"Housing": block, "Shaft": shaft}, root
    )


def nested_assembly(out: Path = FIXTURE_DIR) -> Path:
    """Robot > Arm > Gripper > Finger x2, with translated and rotated levels."""
    finger = Box(10, 4, 20)
    base = Box(40, 40, 10)
    inner = Node(
        "Gripper",
        loc=Pos(0, 0, 30),
        children=[
            Node("FingerL", "Finger", Pos(-8, 0, 0)),
            Node("FingerR", "Finger", Pos(8, 0, 0)),
        ],
    )
    arm = Node("Arm", loc=Pos(100, 0, 0) * Rot(0, 0, 90), children=[inner])
    root = Node("Robot", children=[Node("Base", "Base", Location()), arm])
    return write_assembly(out / "nested_assembly.step", {"Finger": finger, "Base": base}, root)


def generate_assemblies(out: Path = FIXTURE_DIR) -> dict[str, Path]:
    """Write all assembly fixtures; returns name -> path."""
    builders = (two_plates_assembly, tapped_assembly, shaft_in_bearing_assembly, nested_assembly)
    return {fn.__name__: fn(out) for fn in builders}


if __name__ == "__main__":
    for n, p in generate_assemblies().items():
        print(n, p)
