# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Pydantic output contract. Field names are final."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.4.0"

AxisLabel = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- primitives ----------
class Vec3(_M):
    x: float
    y: float
    z: float


class Axis(_M):
    origin: Vec3
    direction: Vec3


class BBox(_M):
    min: Vec3
    max: Vec3
    size: Vec3


class OrientedBBox(_M):
    center: Vec3
    axes: list[Vec3]
    half_sizes: Vec3
    size_sorted: list[float]


class Inference(_M):
    """Mixin for every inferred thing (facts vs. inferences, rule 7)."""

    confidence: float
    evidence: str


# ---------- per-part facts ----------
class MassProperties(_M):
    volume_mm3: float
    surface_area_mm2: float
    centroid: Vec3
    density_g_cm3: float | None = None
    mass_g: float | None = None
    material_assumed: str | None = None
    material_source: Literal["user", "name_guess", "none"] = "none"
    inertia_tensor_g_mm2: list[list[float]] | None = None


class TopologySummary(_M):
    solids: int
    shells: int
    faces: int
    edges: int
    vertices: int
    face_types: dict[str, int]


class StandardMatch(Inference):
    standard: str
    designation: str
    fit: Literal["clearance_close", "clearance_normal", "clearance_loose", "tap_drill", "nominal"]
    nominal_diameter_mm: float
    deviation_mm: float


class HoleSegment(_M):
    kind: Literal["cylinder", "cone"]
    diameter_mm: float
    diameter_small_mm: float | None = None
    cone_angle_deg: float | None = None
    start_depth_mm: float
    end_depth_mm: float
    face_ids: list[str]


class Hole(_M):
    id: str
    axis: Axis
    diameter_mm: float
    depth_mm: float | None
    is_through: bool
    entry_type: Literal["plain", "counterbore", "countersink", "counterbore_and_countersink"]
    counterbore_diameter_mm: float | None = None
    counterbore_depth_mm: float | None = None
    countersink_diameter_mm: float | None = None
    countersink_angle_deg: float | None = None
    bottom_type: Literal["through", "flat", "drill_point", "other"]
    segments: list[HoleSegment]
    standard_matches: list[StandardMatch] = Field(default_factory=list)
    likely_threaded: bool = False
    entry_face_id: str | None = None
    edge_distance_mm: float | None = None
    warnings: list[str] = Field(default_factory=list)


class HolePattern(_M):
    id: str
    kind: Literal["linear", "rectangular_grid", "circular", "irregular_group"]
    hole_ids: list[str]
    count: int
    diameter_mm: float
    normal: Vec3
    pitch_mm: float | None = None
    grid_rows: int | None = None
    grid_cols: int | None = None
    row_pitch_mm: float | None = None
    col_pitch_mm: float | None = None
    center: Vec3 | None = None
    pitch_circle_diameter_mm: float | None = None
    angular_pitch_deg: float | None = None
    description: str


class Fillet(_M):
    id: str
    radius_mm: float
    convex: bool
    face_ids: list[str]
    approx_length_mm: float


class Chamfer(_M):
    id: str
    distance_mm: float
    angle_deg: float
    face_ids: list[str]
    approx_length_mm: float


class Boss(_M):
    id: str
    axis: Axis
    diameter_mm: float
    height_mm: float
    face_ids: list[str]
    has_hole_id: str | None = None


class Slot(_M):
    id: str
    width_mm: float
    length_mm: float
    depth_mm: float | None
    is_through: bool
    axis_dir: Vec3
    center: Vec3
    face_ids: list[str]
    cutout_id: str | None = None  # the canonical cut-out record for the same opening


class Pocket(_M):
    id: str
    depth_mm: float
    outline_size: tuple[float, float]
    floor_face_id: str
    corner_radius_mm: float | None = None
    cutout_id: str | None = None  # the canonical cut-out record for the same opening


class CutoutLevel(_M):
    """One level of a cut-out, outline measured on a plane parallel to the entry face."""

    shape: Literal["circle", "rectangle", "rounded_rectangle", "obround", "polygon", "freeform"]
    width_mm: float
    length_mm: float
    rotation_deg: float  # of the length axis against the part's u axis, in (-90, 90]
    corner_radius_mm: float | None = None
    sides: int | None = None
    center: Vec3  # part frame, on the level's entry plane
    center_uv: tuple[float, float]  # ordinates from the host outline corner (same as holes)
    u_range: tuple[float, float]  # min / max u ordinate of the outline
    v_range: tuple[float, float]
    vertices_uv: list[tuple[float, float]] = Field(default_factory=list)  # corners, max 12
    depth_mm: float | None  # None = through (last level only)
    area_mm2: float
    perimeter_mm: float
    draft_deg: float | None = None
    outline: list[dict[str, Any]] = Field(default_factory=list)  # ordered lines / arcs (u, v)


class Cutout(_M):
    id: str  # "C001"
    kind: Literal["through", "recess", "stepped", "notch"]
    host_face_id: str
    entry_normal: Vec3
    levels: list[CutoutLevel]  # from the entry face inward
    total_depth_mm: float | None  # None if through
    edge_distance_mm: float | None = None
    web_to_neighbor_mm: float | None = None  # thinnest bridge to another opening
    edge_side: str | None = None  # notches: which edge of the outline
    pattern_id: str | None = None
    face_ids: list[str]
    floor_type: Literal["flat", "other", "through"] = "through"
    description: str
    warnings: list[str] = Field(default_factory=list)


class CutoutPattern(_M):
    id: str
    kind: Literal["linear", "rectangular_grid", "irregular_group"]
    cutout_ids: list[str]
    count: int
    pitch_mm: float | None = None
    description: str


class ShapeClass(Inference):
    label: Literal[
        "plate",
        "l_bracket",
        "u_bracket",
        "z_bracket",
        "angle",
        "bar",
        "extrusion_profile",
        "tube",
        "shaft",
        "disc",
        "ring",
        "block",
        "housing",
        "gear_like",
        "sheet_metal",
        "fastener",
        "freeform",
        "other",
    ]
    thickness_mm: float | None = None
    notes: list[str] = Field(default_factory=list)


class SemanticTag(Inference):
    kind: Literal[
        "motor_mount",
        "board_mount",
        "servo_mount",
        "bearing_seat",
        "shaft_bore",
        "extrusion_profile",
        "fastener_hole_pattern",
        "standoff",
        "belt_slot",
        "other",
    ]
    label: str
    knowledge_id: str | None = None
    feature_ids: list[str] = Field(default_factory=list)


# ---------- understanding layer ----------
class Evidence(_M):
    code: str  # machine-readable, e.g. "shape_class:l_bracket", "neighbor:motor", "name:shoulder"
    refs: list[str] = Field(default_factory=list)  # IDs involved (faces, holes, parts, joints)
    weight: float = 0.0
    text: str = ""


class Hypothesis(_M):
    label: str
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)


class AAGSummary(_M):
    nodes: int
    edges: int
    edge_types: dict[str, int]  # convex / concave / smooth
    base_body: str | None = None


class StructuralFeature(_M):
    id: str  # "S001"
    kind: Literal[
        "rib",
        "gusset",
        "web",
        "step",
        "lightening_cutout",
        "bend",
        "flange",
        "standoff_wall",
    ]
    face_ids: list[str]
    thickness_mm: float | None = None
    height_mm: float | None = None
    length_mm: float | None = None
    angle_deg: float | None = None
    radius_mm: float | None = None
    supports: list[str] = Field(default_factory=list)
    description: str
    confidence: float


class SymmetryInfo(_M):
    mirror_planes: list[dict[str, Any]] = Field(default_factory=list)
    rotational: list[dict[str, Any]] = Field(default_factory=list)
    description: str = ""


class ProcessGuess(_M):
    ranked: list[Hypothesis]
    source: Literal["inferred", "user"] = "inferred"


class PartUnderstanding(_M):
    aag: AAGSummary
    structural_features: list[StructuralFeature] = Field(default_factory=list)
    symmetry: SymmetryInfo
    process: ProcessGuess
    roles: list[Hypothesis] = Field(default_factory=list)
    mirror_of_part_id: str | None = None


class RigidGroup(_M):
    id: str  # "L0" ground, "L1", ...
    instance_ids: list[str]
    is_ground: bool = False
    ground_evidence: list[Evidence] = Field(default_factory=list)
    mass_g: float | None = None
    description: str = ""


class KinJoint(_M):
    id: str  # "KJ1", ordered from ground outward
    kind: Literal["revolute", "prismatic", "cylindrical", "fixed_uncertain"]
    parent_link: str
    child_link: str
    axis: Axis
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float
    driven_by: str | None = None
    range_deg_or_mm: tuple[float, float] | None = None
    blocked_by: list[str] = Field(default_factory=list)
    description: str = ""


class Mechanism(_M):
    id: str  # "M1"
    kind: Literal[
        "gear_pair",
        "belt_drive",
        "lead_screw",
        "direct_drive",
        "rack_and_pinion",
        "linkage_loop",
        "unknown",
    ]
    instance_ids: list[str]
    ratio: float | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_instance_id: str | None = None
    output_joint_id: str | None = None
    confidence: float
    evidence: list[Evidence] = Field(default_factory=list)
    description: str = ""


class KinematicModel(_M):
    links: list[RigidGroup]
    joints: list[KinJoint]
    mechanisms: list[Mechanism] = Field(default_factory=list)
    dof: int = 0
    topology: Literal[
        "static",
        "serial_chain",
        "tree",
        "closed_loop",
        "mobile_base",
        "mobile_manipulator",
        "mixed",
    ] = "static"
    floating_groups: list[str] = Field(default_factory=list)
    description: str = ""
    mermaid: str = ""


class LoadPath(_M):
    id: str
    load_source: str
    path: list[str]
    weakest_connection: str | None = None
    fastener_count_min: int | None = None
    description: str = ""


class WeakSpot(_M):
    id: str  # "W001"
    severity: Literal["high", "medium", "low", "info"]
    category: Literal[
        "edge_distance",
        "cutout_web",
        "thin_wall",
        "sharp_internal_corner",
        "cantilever",
        "single_fastener",
        "short_screw",
        "unsupported_span",
        "floating_part",
        "tipping",
        "interference",
        "unfilleted_bend",
        "joint_play",
        "other",
    ]
    refs: list[str] = Field(default_factory=list)
    value: float | None = None
    threshold: float | None = None
    unit: str | None = None
    message: str
    suggestion: str | None = None
    depends_on_assumption: str | None = None


class StabilityInfo(_M):
    center_of_mass: Vec3
    support_points: list[Vec3] = Field(default_factory=list)
    support_polygon_area_mm2: float | None = None
    com_height_mm: float
    min_margin_mm: float | None = None
    tipping_angle_deg: float | None = None
    tipping_direction: Vec3 | None = None
    extended_center_of_mass: Vec3 | None = None
    extended_margin_mm: float | None = None
    extended_tipping_angle_deg: float | None = None
    description: str = ""


class SubassemblyRole(_M):
    subassembly: str
    roles: list[Hypothesis]


class AnswerOption(_M):
    value: str
    label: str


class Question(_M):
    id: str  # stable across runs: hash of (kind, refs)
    kind: Literal[
        "confirm_role",
        "choose_role",
        "material",
        "process",
        "connection_type",
        "joint_drive",
        "joint_range",
        "payload",
        "purpose",
        "load",
        "environment",
        "missing_part",
        "free_text",
    ]
    priority: int
    text: str
    refs: list[str] = Field(default_factory=list)
    why: str
    answer_type: Literal[
        "yes_no", "single_choice", "multi_choice", "number", "number_with_unit", "text"
    ] = "text"
    options: list[AnswerOption] | None = None
    unit: str | None = None
    default_guess: str | None = None
    image: str | None = None
    applies_to: list[str] = Field(default_factory=list)


class Answer(_M):
    question_id: str
    status: Literal["answered", "skipped", "not_sure"]
    value: str | float | list[str] | None = None
    note: str | None = None
    answered_at: str | None = None
    question: str | None = None  # the question text, so the pack can print a Q&A list


class Understanding(_M):
    kinematics: KinematicModel
    load_paths: list[LoadPath] = Field(default_factory=list)
    weak_spots: list[WeakSpot] = Field(default_factory=list)
    stability: StabilityInfo | None = None
    subassembly_roles: list[SubassemblyRole] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    answers: list[Answer] = Field(default_factory=list)
    summary: str = ""


class Part(_M):
    id: str
    name: str
    source_file: str
    content_hash: str
    is_valid_solid: bool
    bbox: BBox
    obb: OrientedBBox
    mass: MassProperties
    topology: TopologySummary
    shape_class: ShapeClass
    holes: list[Hole] = Field(default_factory=list)
    hole_patterns: list[HolePattern] = Field(default_factory=list)
    fillets: list[Fillet] = Field(default_factory=list)
    chamfers: list[Chamfer] = Field(default_factory=list)
    bosses: list[Boss] = Field(default_factory=list)
    slots: list[Slot] = Field(default_factory=list)
    pockets: list[Pocket] = Field(default_factory=list)
    cutouts: list[Cutout] = Field(default_factory=list)
    cutout_patterns: list[CutoutPattern] = Field(default_factory=list)
    semantic_tags: list[SemanticTag] = Field(default_factory=list)
    likely_purchased_hardware: bool = False
    hardware_guess: str | None = None
    color_rgb: list[float] | None = None
    min_wall_thickness_mm: float | None = None
    images: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    understanding: PartUnderstanding | None = None


# ---------- assembly ----------
class Instance(_M):
    id: str
    part_id: str
    parent_instance_id: str | None
    path: str
    position: Vec3
    rotation_matrix: list[list[float]]
    rotation_axis_angle: tuple[Vec3, float]
    global_bbox: BBox
    subassembly: str | None = None


class BOMLine(_M):
    part_id: str
    name: str
    quantity: int
    mass_each_g: float | None = None
    category: Literal["fabricated", "purchased_hardware", "unknown"]


class Contact(_M):
    id: str
    instance_a: str
    instance_b: str
    kind: Literal["planar_face", "cylindrical_fit", "line", "point", "near_miss"]
    min_distance_mm: float
    contact_area_mm2: float | None = None
    normal: Vec3 | None = None
    fit: Literal["clearance", "line_to_line", "interference"] | None = None
    fit_value_mm: float | None = None


class FastenerJoint(Inference):
    id: str
    instance_ids: list[str]
    hole_refs: list[tuple[str, str]]
    axis: Axis
    common_diameter_mm: float
    stack_thickness_mm: float
    suggested_fastener: str | None = None
    has_threaded_end: bool = False
    existing_fastener_instance_id: str | None = None


class Relation(_M):
    id: str
    subject: str
    predicate: Literal[
        "rests_on",
        "bolted_to",
        "mounted_on",
        "inserted_into",
        "adjacent_to",
        "contains",
        "aligned_with",
    ]
    object: str
    via: list[str]
    sentence: str
    confidence: float


class SpatialFact(_M):
    subject: str
    relation: Literal[
        "above", "below", "left_of", "right_of", "in_front_of", "behind", "inside", "centered_on"
    ]
    object: str
    distance_mm: float | None = None
    sentence: str


class Subassembly(_M):
    name: str
    instance_ids: list[str]
    bbox: BBox
    mass_g: float | None = None
    summary: str


class AssemblyInfo(_M):
    root_name: str
    up_axis: AxisLabel = "+Z"
    front_axis: AxisLabel = "-Y"
    instances: list[Instance]
    subassemblies: list[Subassembly]
    bom: list[BOMLine]
    global_bbox: BBox
    total_mass_g: float | None = None
    center_of_mass: Vec3 | None = None
    contacts: list[Contact] = Field(default_factory=list)
    fastener_joints: list[FastenerJoint] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    spatial_facts: list[SpatialFact] = Field(default_factory=list)
    interferences: list[dict[str, Any]] = Field(default_factory=list)
    fastener_shopping_list: list[dict[str, Any]] = Field(default_factory=list)


class Meta(_M):
    schema_version: str
    tool_version: str
    generated_at: str | None
    source_files: list[str]
    original_length_unit: str
    tolerances: dict[str, float]
    options: dict[str, Any]


class Report(_M):
    meta: Meta
    parts: list[Part]
    assembly: AssemblyInfo | None = None
    understanding: Understanding | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


def json_schema() -> dict[str, Any]:
    """JSON Schema of the Report contract."""
    return Report.model_json_schema()
