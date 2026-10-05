# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""XCAF-based STEP reading: names, colors, assembly tree and units."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.IFSelect import IFSelect_RetDone
from OCP.OCP.collections import Sequence_TDF_Label
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label, TDF_Tool
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Shape
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import (
    XCAFDoc_ColorTool,
    XCAFDoc_ColorType,
    XCAFDoc_DocumentTool,
    XCAFDoc_ShapeTool,
)


class StepReadError(Exception):
    """Raised when a STEP file cannot be read at all."""


@dataclass
class ProtoPart:
    """One unique prototype shape (analysed once, in its local frame)."""

    key: str
    name: str
    shape: TopoDS_Shape
    color_rgb: list[float] | None = None


@dataclass
class InstanceRec:
    """One placement of a prototype in the assembly."""

    proto_key: str
    path: str
    matrix: np.ndarray  # 4x4 global transform (rigid)
    parent_path: str | None
    subassembly: str | None


@dataclass
class StepModel:
    """Result of reading one STEP file."""

    path: str
    original_unit: str
    root_name: str
    protos: dict[str, ProtoPart] = field(default_factory=dict)
    instances: list[InstanceRec] = field(default_factory=list)
    is_assembly: bool = False
    warnings: list[str] = field(default_factory=list)


_UNIT_PREFIX = {
    ".MILLI.": "mm",
    ".CENTI.": "cm",
    ".KILO.": "km",
    "$": "m",
}


def detect_original_unit(path: str | Path) -> str:
    """Original length unit by scanning the STEP text (OCCT converts to mm itself)."""
    text = Path(path).read_text(encoding="latin-1", errors="ignore")
    m = re.search(r"CONVERSION_BASED_UNIT\s*\(\s*'([A-Za-z ]+)'", text)
    if m:
        name = m.group(1).strip().lower()
        return {"inch": "in", "foot": "ft"}.get(name, name)
    for m in re.finditer(
        r"LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\(\s*\*\s*\)\s*SI_UNIT\s*\(([^)]*)\)", text
    ):
        prefix = m.group(1).split(",")[0].strip()
        return _UNIT_PREFIX.get(prefix, "mm")
    m = re.search(r"SI_UNIT\s*\(\s*(\.\w+\.|\$)\s*,\s*\.METRE\.", text)
    if m:
        return _UNIT_PREFIX.get(m.group(1), "mm")
    return "mm"


def _label_name(label: TDF_Label) -> str | None:
    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        text = str(attr.Get().ToExtString()).strip()
        return text or None
    return None


def _label_key(label: TDF_Label) -> str:
    entry = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, entry)
    return str(entry.ToCString())


def _label_color(color_tool: XCAFDoc_ColorTool, label: TDF_Label) -> list[float] | None:
    for kind in (XCAFDoc_ColorType.XCAFDoc_ColorSurf, XCAFDoc_ColorType.XCAFDoc_ColorGen):
        c = Quantity_Color()
        if color_tool.GetColor_s(label, kind, c):
            return [round(c.Red(), 4), round(c.Green(), 4), round(c.Blue(), 4)]
    return None


def _matrix(loc: TopLoc_Location) -> np.ndarray:
    t = loc.Transformation()
    m = np.eye(4)
    for r in range(3):
        for c in range(4):
            m[r, c] = t.Value(r + 1, c + 1)
    return m


def _seq(seq: Sequence_TDF_Label) -> list[TDF_Label]:
    return [seq.Value(i) for i in range(1, seq.Length() + 1)]


_GENERIC_NAMES = {"", "compound", "none", "solid", "shape", "part", "open cascade step translator"}


def _is_generic(name: str | None) -> bool:
    return (
        name is None
        or name.strip().lower().startswith("open cascade")
        or name.strip().lower() in _GENERIC_NAMES
    )


class _Walker:
    def __init__(
        self, shape_tool: XCAFDoc_ShapeTool, color_tool: XCAFDoc_ColorTool, model: StepModel
    ):
        self.st = shape_tool
        self.ct = color_tool
        self.model = model
        self.counter = 0

    def proto(self, label: TDF_Label) -> ProtoPart:
        key = _label_key(label)
        if key not in self.model.protos:
            self.counter += 1
            raw = _label_name(label)
            stem = Path(self.model.path).stem
            name = (
                stem
                if self.counter == 1 and _is_generic(raw)
                else (f"{stem}_{self.counter}" if _is_generic(raw) else str(raw))
            )
            shape = XCAFDoc_ShapeTool.GetShape_s(label)
            self.model.protos[key] = ProtoPart(key, name, shape, _label_color(self.ct, label))
        return self.model.protos[key]

    def walk(
        self, label: TDF_Label, loc: TopLoc_Location, path: str, parent: str | None, sub: str | None
    ) -> None:
        st = XCAFDoc_ShapeTool
        if st.IsAssembly_s(label):
            self.model.is_assembly = True
            name = _label_name(label) or "Assembly"
            here = f"{path}/{name}" if path else name
            comps = Sequence_TDF_Label()
            st.GetComponents_s(label, comps)
            for comp in _seq(comps):
                self.walk(comp, loc, here, here, sub)
        elif st.IsReference_s(label):
            ref = TDF_Label()
            st.GetReferredShape_s(label, ref)
            new_loc = loc.Multiplied(st.GetLocation_s(label))
            ref_name = _label_name(label) or _label_name(ref) or "Part"
            if st.IsAssembly_s(ref):
                self.model.is_assembly = True
                here = f"{path}/{ref_name}" if path else ref_name
                comps = Sequence_TDF_Label()
                st.GetComponents_s(ref, comps)
                sub_name = sub or ref_name
                for comp in _seq(comps):
                    self.walk(comp, new_loc, here, here, sub_name)
            else:
                proto = self.proto(ref)
                inst_path = f"{path}/{ref_name}" if path else ref_name
                self._add(proto, new_loc, inst_path, parent, sub)
        else:
            proto = self.proto(label)
            inst_path = f"{path}/{proto.name}" if path else proto.name
            self._add(proto, loc, inst_path, parent, sub)

    def _add(
        self, proto: ProtoPart, loc: TopLoc_Location, path: str, parent: str | None, sub: str | None
    ) -> None:
        n = sum(1 for i in self.model.instances if i.path.rsplit(":", 1)[0] == path)
        full = f"{path}:{n + 1}" if n else path
        self.model.instances.append(InstanceRec(proto.key, full, _matrix(loc), parent, sub))


def read_step(path: str | Path) -> StepModel:
    """Read a STEP file into prototypes + instances. Lengths are converted to mm by OCCT."""
    p = Path(path)
    if not p.is_file():
        raise StepReadError(f"file not found: {p}")
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    app.InitDocument(doc)
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)
    if reader.ReadFile(str(p)) != IFSelect_RetDone:
        raise StepReadError(f"cannot parse STEP file: {p.name}")
    if not reader.Transfer(doc):
        raise StepReadError(f"no geometry could be transferred from {p.name}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    model = StepModel(str(p), detect_original_unit(p), p.stem)
    free = Sequence_TDF_Label()
    shape_tool.GetFreeShapes(free)
    walker = _Walker(shape_tool, color_tool, model)
    labels = _seq(free)
    for label in labels:
        walker.walk(label, TopLoc_Location(), "", None, None)
    if len(labels) == 1:
        raw = _label_name(labels[0])
        model.root_name = p.stem if _is_generic(raw) else str(raw)
    if not model.instances:
        raise StepReadError(f"no shapes found in {p.name}")
    _drop_empty_prototypes(model, p.name)
    return model


def _drop_empty_prototypes(model: StepModel, filename: str) -> None:
    """Remove prototypes without any face (assemblies that reference other STEP files)."""
    from OCP.TopAbs import TopAbs_FACE

    from stepscribe.geometry.occ_utils import unique_subshapes

    empty = [
        k for k, proto in model.protos.items() if not unique_subshapes(proto.shape, TopAbs_FACE)
    ]
    if not empty:
        return
    if len(empty) == len(model.protos):
        raise StepReadError(
            f"{filename} has assembly structure but no geometry: its parts are external references "
            "to other STEP files. Put those files next to it, or export a single self-contained STEP."
        )
    names = ", ".join(sorted(model.protos[k].name for k in empty))
    model.warnings.append(f"{len(empty)} part(s) without geometry were skipped: {names}")
    for k in empty:
        del model.protos[k]
    model.instances = [i for i in model.instances if i.proto_key in model.protos]
