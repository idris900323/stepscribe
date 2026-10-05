# PRT001 plate_4xM3

- Source file: plate_4xM3.step
- Size (oriented bounding box): 60.0 × 60.0 × 5.0 mm
- Axis-aligned size: 60.0 × 60.0 × 5.0 mm
- Volume: 17818.4 mm³; surface area: 8541.0 mm²
- Mass: 48.1 g (material alu (given))
- Valid closed solid: yes
- Topology: 10 faces, 24 edges (cylinder 4, plane 6)
- Shape class: Likely: plate (0.90): thickness 5.0 < 0.2 × 60.0; 83% of area on two parallel planes
- Thickness: 5.0 mm
- Minimum wall thickness (sampled estimate): 5.00 mm

## Recognised standards and features
- Likely: 4 × M3 clearance normal holes (Ø3.4 THRU) (0.90): 4 × Ø3.4 THRU, rectangular 40.0 × 40.0 (2 × 2); Ø3.40 vs ISO 273 medium Ø3.40 for M3 (dev +0.00 mm)

## Holes
| ID | Ø | Depth | Entry | Bottom | Position (x, y, z) | Direction | Edge distance | Likely standard |
|---|---|---|---|---|---|---|---|---|
| H001 | 3.40 | THRU | plain | through | (-20.00, -20.00, 2.50) | (0.00, 0.00, -1.00) | 8.30 | Likely: M3 clearance normal (1.00) |
| H002 | 3.40 | THRU | plain | through | (-20.00, 20.00, 2.50) | (0.00, 0.00, -1.00) | 8.30 | Likely: M3 clearance normal (1.00) |
| H003 | 3.40 | THRU | plain | through | (20.00, -20.00, 2.50) | (0.00, 0.00, -1.00) | 8.30 | Likely: M3 clearance normal (1.00) |
| H004 | 3.40 | THRU | plain | through | (20.00, 20.00, 2.50) | (0.00, 0.00, -1.00) | 8.30 | Likely: M3 clearance normal (1.00) |

## Hole patterns
- 4 × Ø3.4 THRU, rectangular 40.0 × 40.0 (2 × 2) (P001; holes H001, H002, H003, H004)

## Reconstruction (measured build recipe, part's own coordinates, mm)
- Oriented box: centre (0.00, 0.00, 0.00), half sizes (30.00, 30.00, 2.50) along axes (0.000, -1.000, 0.000); (1.000, 0.000, 0.000); (0.000, 0.000, 1.000).
- Axis-aligned bounds: min (-30.00, -30.00, -2.50), max (30.00, 30.00, 2.50).
- Main face: the largest flat face (3563.7 mm², normal (0.000, 0.000, 1.000)).
- Frame: u = (0.000, 1.000, 0.000), v = (-1.000, 0.000, 0.000), w = (0.000, 0.000, 1.000) (w points from the main face into the part).
- The part extends 5.00 mm along w. Origin of (u, v) = lowest-u, lowest-v corner of the outline; w = 0 on the main face.
- Outer outline, 4 segment(s), listed in order, (u, v) in mm:
-   1. line (0.00, 60.00) -> (60.00, 60.00) (length 60.00)
-   2. line (60.00, 60.00) -> (60.00, 0.00) (length 60.00)
-   3. line (60.00, 0.00) -> (0.00, 0.00) (length 60.00)
-   4. line (0.00, 0.00) -> (0.00, 60.00) (length 60.00)
- 4 circular opening(s) in the main face: see the hole table (positions below).
- Feature positions in this frame (u, v, w in mm; hole positions are where the hole enters the part):
-   - H001 hole dia 3.40 at (u 10.00, v 50.00, w 5.00)
-   - H002 hole dia 3.40 at (u 50.00, v 50.00, w 5.00)
-   - H003 hole dia 3.40 at (u 10.00, v 10.00, w 5.00)
-   - H004 hole dia 3.40 at (u 50.00, v 10.00, w 5.00)
- Largest 10 of 10 faces, positions (a, b, c) in mm along the oriented-box axes listed above, measured from the box corner that has the lowest a, b and c:
-   - F0007 plane, area 3563.7, normal (0.00, 0.00, 1.00), centre (30.00, 30.00, 0.00)
-   - F0008 plane, area 3563.7, normal (0.00, 0.00, 1.00), centre (30.00, 30.00, 5.00)
-   - F0005 plane, area 300.0, normal (1.00, 0.00, 0.00), centre (30.00, 0.00, 2.50)
-   - F0006 plane, area 300.0, normal (0.00, 1.00, 0.00), centre (60.00, 30.00, 2.50)
-   - F0009 plane, area 300.0, normal (0.00, 1.00, 0.00), centre (0.00, 30.00, 2.50)
-   - F0010 plane, area 300.0, normal (1.00, 0.00, 0.00), centre (30.00, 60.00, 2.50)
-   - F0001 cylinder R1.70, area 53.4, axis (0.00, 0.00, 1.00), centre (50.00, 10.00, 2.50)
-   - F0002 cylinder R1.70, area 53.4, axis (0.00, 0.00, 1.00), centre (10.00, 10.00, 2.50)
-   - F0003 cylinder R1.70, area 53.4, axis (0.00, 0.00, 1.00), centre (50.00, 50.00, 2.50)
-   - F0004 cylinder R1.70, area 53.4, axis (0.00, 0.00, 1.00), centre (10.00, 50.00, 2.50)
