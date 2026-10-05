# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 idris sadiq
"""Central tolerances and defaults. Rule 5: no magic numbers elsewhere."""

from __future__ import annotations

# --- tolerances -----------------------------------------------------------
LINEAR_TOL = 1e-3  # mm, general geometric equality
ANGULAR_TOL_DEG = 0.5  # degrees, parallel / coaxial tests
COAXIAL_TOL = 0.01  # mm, max distance between two axis lines
CONTACT_TOL = 0.05  # mm, assembly contact threshold
NEAR_MISS_MAX = 1.0  # mm
AABB_INFLATE = 0.1  # mm, broad-phase inflation
FULL_BORE_MIN_SPAN_DEG = 359.0  # a bore must wrap at least this much
MIN_FEATURE_SIZE = 0.1  # mm, features below this are ignored with a warning
PATTERN_DIAMETER_TOL = 0.01  # mm
PATTERN_FIT_TOL = 0.01  # mm, circle fit / lattice residual
PATTERN_ANGLE_TOL_DEG = 0.1
HOLE_CLASSIFY_OFFSET_MIN = 0.05  # mm, probe offset beyond hole ends
HOLE_CLASSIFY_OFFSET_REL = 0.01  # of diameter
BORE_PROBE_REL = 0.1  # of radius
BORE_PROBE_MAX = 0.05  # mm
CSK_SNAP_ANGLES = (82.0, 90.0, 100.0, 120.0)
CSK_SNAP_TOL_DEG = 1.0
DRILL_POINT_ANGLE = 118.0
DRILL_POINT_TOL_DEG = 3.0
STANDARD_MATCH_SCALE = 0.15  # mm: confidence = 1 - |dev| / scale
STANDARD_MATCH_MIN_CONF = 0.3
STANDARD_MATCH_TOP_N = 3
CBORE_MATCH_BOOST = 0.2
CBORE_MATCH_TOL = 0.3  # mm
THREAD_TAP_CONF = 0.7
THREAD_EXACT_TOL = 1e-3

# --- sampling -------------------------------------------------------------
WALL_SAMPLES = 200
WALL_SEED = 12345
MESH_DEFLECTION_REL = 0.001
MESH_ANGULAR_DEFLECTION = 0.3

# --- output ---------------------------------------------------------------
FLOAT_PRECISION = 4
DEFAULT_BUDGET_TOKENS = 40000
CHARS_PER_TOKEN = 4
DEFAULT_TIMEOUT_S = 300

# --- materials (g/cm3) ----------------------------------------------------
MATERIAL_KEYWORDS: dict[str, float] = {
    "alu": 2.70,
    "steel": 7.85,
    "petg": 1.27,
    "pla": 1.24,
    "abs": 1.04,
    "nylon": 1.01,
    "pa12": 1.01,
    "cf": 1.60,
    "brass": 8.50,
}

# --- features ---------------------------------------------------------------
TANGENT_TOL_DEG = 2.0  # G1 test between neighbouring faces
SLOT_SPAN_DEG = 180.0
SLOT_SPAN_TOL_DEG = 2.0
CHAMFER_MIN_ANGLE_DEG = 20.0  # chamfer normal vs neighbour normal
CHAMFER_MAX_ANGLE_DEG = 70.0
CHAMFER_MAX_WIDTH_REL = 0.1  # of part diagonal
CHAMFER_MAX_ASPECT = 0.4  # width / length of the strip
BOSS_MAX_HEIGHT_RATIO = 1.5  # taller convex cylinders are shafts, not bosses
BOSS_BASE_AREA_FACTOR = 1.2  # base plane must exceed the boss footprint by this factor
EDGE_DISTANCE_MAX_FACES = 600
PERCENTILE_WALL = 5.0
CHAMFER_LONG_EDGE_REL = 0.6  # neighbours along the strip's long edges
OBB_MESH_DEFLECTION_REL = 1e-4  # of bbox diagonal; OBB needs a triangulation
OBB_MESH_DEFLECTION_MIN = 1e-3  # mm
PARTIAL_BORE_MIN_SPAN_DEG = (
    190.0  # concave cylinders wrapping more than a slot end still count as holes
)

# --- cut-outs (openings, recesses, notches) ----------------------------------
CUTOUT_HOST_AREA_REL = 0.05  # host faces: planar faces at least this share of the largest one
CUTOUT_FLAT_DOT = 0.995  # |n . n_host| above this: face parallel to the host
CUTOUT_WALL_DOT = 0.1  # |n . n_host| below this: wall (about 5.7 deg of draft allowed)
CUTOUT_AXIS_DOT = 0.995  # cylinder axis parallel to the entry normal
CUTOUT_RIGHT_ANGLE_TOL_DEG = 0.5  # corners of a rectangle
CUTOUT_RADIUS_TOL = 1e-3  # mm, equal corner radii
CUTOUT_CHAIN_TOL = 1e-3  # mm, joining section edges into a loop
CUTOUT_ARC_STEP_DEG = 3.0  # polyline sampling of arcs for hulls and distances
CUTOUT_NOTCH_MIN_DEPTH = 0.5  # mm, an inward run of the outline counts as a notch
CUTOUT_NOTCH_MAX_REL = 0.5  # notch width and depth at most this share of the outline box
CUTOUT_NOTCH_MAX_EDGES = 12
CUTOUT_MAX_NOTCHES = 12
CUTOUT_MIN_STEP = 0.01  # mm, level breakpoints closer than this are one
CUTOUT_MAX_VERTICES = 12
