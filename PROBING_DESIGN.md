# Probing & the Workpiece Frame

Status: **design / seam sketched** (no behavior change to `realWorldGcodeSender.py` yet)

## The idea

Vision and probing own **different axes**, because of what each can physically
measure:

- **X / Y and in-plane angle** come from **vision**. The ChArUco homography maps
  the *bed plane*, so where the stock sits and how it's rotated is recoverable
  from the camera. This is the existing overlay/placement workflow.
- **Z is out of that plane.** A top-down homography can't measure height without
  parallax, so the real top-of-stock -- the thing that sets cut depth -- is
  irreducibly a **probing** problem.

Probing isn't limited to Z, though: touching the **real workpiece edges** turns
vision's *approximate* XY into **concrete, workpiece-oriented XY actuals**. The
difference from the old reference-plate routine is what *seeds* it -- vision
already knows where the edge is to within a pixel, so the probe refines rather
than blindly hunts.

### One spine: the workpiece frame

Everything above collapses into a single object:

> a **workpiece frame** -- XY origin, in-plane angle, and a Z surface -- where
> every value is independently sourced (vision / manual / probe), each tagged
> with its fidelity. The toolpath and the Z-mesh both ride in this frame.

**Probing is optional.** The frame is always complete and cuttable from the
cheapest sources -- eyeballed vision XY + a typed/assumed thickness. Probing only
raises fidelity on whatever you point it at; nothing third-party is ever on the
critical path to a cut. That is exactly the relationship a *plugin* should have
with the core of a machine that moves a spinning cutter.

The pattern is **vision proposes, probe disposes**: vision picks safe probe
targets (on the stock, off the cut path); a probe strategy measures and writes
refined values back into the frame.

## Decision (maintainability)

- **One codebase.** Build this in the running monolith's world, *not* the
  half-built modular `machine/` tree (which has no `grbl.py` and can't drive a
  machine). The duplicate-tree is the real maintainability risk.
- **Extract only the seam that varies.** The valuable abstraction -- the frame
  plus probe *strategies* -- is also the cheapest and most testable part: pure
  data + small functions over a 4-method machine protocol.
- **Portable by design.** `workpiece_frame.py` and `probing/` depend only on a
  tiny `Machine` protocol that the existing `GCodeSender` already satisfies. They
  can be lifted unchanged into `machine/operations/probe.py` later if the modular
  tree ever becomes real. We are not choosing monolith *over* modular -- we're
  building the one piece portable to either.
- **"Plugins" = a registry, not a loader.** Strategies register via a decorator
  into a dict (`@register("z_mesh")`). No dynamic file loading -- safer and more
  debuggable for machine control. Promote to real plugins only when a genuine
  third party exists.

## What's here now

| File | Role |
|------|------|
| `workpiece_frame.py` | Pure data: `Source`, `Measured`, `ZSurface`, `WorkpieceFrame` (+ workpiece->machine transform). No hardware deps. |
| `probing/base.py` | The seam: `Machine` protocol, `ProbeTarget`, `ProbeStrategy`, `register`/`get_strategy`/`available`. |
| `probing/strategies.py` | `z_touch_off` and `z_mesh` (working), `edge_refine` (stub). |
| `probing/mesh.py` | `interpolate_z` -- inverse-distance for now; Delaunay/barycentric later. |

Adapting the monolith is a few lines (not done yet): wrap `GCodeSender` so
`probe / move / position` map to its existing `probe / work_offset_move /
get_absolute_pos`, build a frame from the overlay's offset/rotation, hand it to a
strategy.

## Open / not done

- **Z-mesh -> toolpath warp** (the real algorithmic piece): triangulate samples,
  look up surface Z per cut segment, split long moves so they follow the surface.
  Lives in the path layer, not in `probing/`.
- **`edge_refine` geometry**: mapping touched edges back onto origin + angle,
  one- vs two-edge corner cases, cutter-radius offset at the tool tip.
- **Wiring into `realWorldGcodeSender.py`** (the `z` key path).
- **Vision-driven target selection** (auto-pick safe probe points from the placed
  toolpath).

## Future source: dense 3D capture

The end-game is structured light (or photogrammetry / a laser line scanner):
capture the *actual* surface densely and map it straight onto the mesh. The
design already accommodates this -- `ZSurface.samples` and `interpolate_z()` do
not care whether points came from a touch probe (sparse, very accurate) or a
scan (dense, needs calibration). A scan is just another `Source` that populates
the same mesh, consumed unchanged by `z_at()` and the toolpath warp. So the
cheap touch-probe path being built now and the expensive scan path later **share
the entire downstream** -- no rework when the parts/budget show up.

A concrete, buildable version of this -- a DIY servo-steered laser-line scanner --
is specced in [Appendix A](#appendix-a-diy-laser-line-scanner-the-side-project).

## Explicitly out of scope

The broader `REFACTORING_PLAN.md` (Qt/Web UI, 3D, protobuf calibration, dynamic
plugin manager, etc.). Not pursued here; the modular tree is left in place but
not extended.

---

# Appendix A: DIY laser-line scanner (the side project)

A buildable path to a dense `Source.SCAN`, recorded so the framing -- especially
the optics and the ICP framing -- is right from the start. **None of this is
built; it is a spec.** Everything here lives *behind* the `Source` boundary: it
produces confidence-weighted points into `ZSurface.samples` and nothing
downstream (`z_at()`, the toolpath warp) changes.

## Principle

Laser-line profilometry where **the machine's motion is the scan axis** -- the
CNC is already a precision linear stage that reports position to the thou, so a
~$10 line laser on the tool head + the existing overhead camera gets you a
scanner. A line laser projects a plane of light; where it hits the surface the
camera sees a bright stripe; the stripe's lateral shift from its Z-reference
position encodes height, scaled by the triangulation angle. Each gantry step ->
one profile; stack them -> point cloud -> mesh.

## Calibration: the homography is NOT enough

The ChArUco homography maps the *bed plane* and solves XY registration -- the
expensive part of any scanner -- but it says nothing reliable about height when
the camera is off-center and non-telecentric (every point viewed at a different
oblique angle, so triangulation gain and foreshortening vary across the FOV).
Height triangulation needs the camera's full **intrinsics + extrinsics**
(calibrated pinhole + distortion + 3D pose), recoverable from the same ChArUco
board. Plus a one-time **laser-plane calibration** (plane orientation, and for
the servo rig, orientation *as a function of commanded servo angle*).

## Servo-steered azimuth

Mount the laser at a **fixed tilt** on a servo that **yaws it about the tool
axis**. Rotating the servo sweeps the incidence azimuth around a *cone*, so N
servo stops = N illumination directions -- full 360deg azimuth diversity from one
actuator. This is the hardware realization of cross-azimuth consensus (below).

Two motion "directions" do **different jobs** -- keep them separate:

- **Forward vs reverse sweep (same orientation)**: illumination + view are
  identical at each point (rigid laser, pure translation), so this is an
  **accuracy** axis only -- cancels backlash / motion lag, averages noise, and
  forward/reverse disagreement is a confidence/QA signal. It does **not** fill
  shadows.
- **Azimuth / orientation (servo angle)**: changes incidence direction, so this
  is the **coverage + photometric** axis -- fills geometric occlusion *and*
  samples different optical conditions.

What's unreachable after a couple of azimuths (undercuts, deep narrow pockets)
is also unreachable by the endmill, so coverage stops where the cutter does.

## Optics: surfaces are not Lambertian, the camera is not telecentric

Geometric occlusion is azimuth-dependent but direction-invariant. The
**photometric** stripe-center, though, is biased by the real BRDF and that bias
is **directional and depends on the material and its surroundings**:

- specular glints displaced from the true center (shiny metal, finished/wet wood)
- subsurface scatter / translucency shifting the apparent center (plastics, wet
  wood, clear acrylic, where refraction also bends the stripe)
- **interreflections** -- a nearby wall or shiny clamp bouncing a second copy of
  the line onto the point

This is *systematic* error, not noise, so you cannot average it away. But it is
**direction-specific while geometry is not** -- so multi-azimuth *consensus*
identifies truth and *disagreement* flags artifacts. Mitigations at the source:
**cross-polarization** (polarizer on laser + crossed analyzer on lens kills most
specular), a **bandpass filter** at the laser wavelength, exposure bracketing,
and **peak-shape vetting** (clean symmetric Gaussian = trustworthy; skewed /
double-peaked = specular/subsurface suspect -> a free per-sample confidence).

## Per-sample confidence

Every scanned point carries a confidence ~ (stripe peak quality) x (cross-azimuth
agreement). This drives robust fusion (below) and the fallback decision.

## Process isolation

The scanner runs as a **separate process** (heavy CV + CPU-bound, so a real
process parallelizes where threads would not under the GIL; crash isolation so a
camera/driver fault never takes down the machine controller; standalone
build/test against a known object). Hard rule: **single owner of the GRBL
serial** -- the main app owns the machine connection; the scanner owns only the
camera and its own laser-servo MCU and **requests gantry moves over a narrow
contract** ("scan this region with these params" -> progress + confidence-
weighted cloud). That contract *is* the `Source` boundary.

**Step-and-shoot** the two motion systems (gantry move, stop; servo to azimuth,
stop, capture; repeat) to eliminate sync entirely -- speed does not matter.

Enables **adaptive azimuths**: coarse pass first, then spend extra servo angles
only on low-confidence regions instead of brute-forcing N passes everywhere.

## Fusing the passes: ICP framing (the part to get right)

You have a **strong pose prior** (machine XY + commanded servo angle + calibrated
camera), so passes are already nearly co-registered. Therefore:

- **Do NOT use free 6-DOF point-to-point ICP per pass.** Every pass images the
  *same static surface* under different light -- there is no rigid motion to
  recover. Unconstrained ICP "explains" systematic measurement error (servo slop,
  calibration, photometric bias) as a pose shift and slides clouds together,
  **laundering bias into a bogus transform and smearing features**. ICP that
  converges by hiding error is the trap.
- **Treat residual misalignment as calibration, not registration.** Use the
  multi-azimuth overlap to estimate a small *shared* correction (servo-zero
  offset, laser-tilt bias) across all passes -- a constrained mini bundle
  adjustment, not a free transform per pass. Coupling: a cheaper/open-loop servo
  has more angle error to recover *and* more room to absorb photometric bias, so
  constraining the DOF matters more, not less.
- **Point-to-plane, robust, confidence-weighted.** Height fields converge far
  better point-to-plane; trimmed/Tukey weights driven by per-sample confidence
  survive specular outliers; reject low-confidence correspondences.
- **The strong use of registration is scan -> probe anchoring.** Probe points are
  accurate/sparse, the scan is dense/biased. Register the scan onto the *fixed*
  touch-probe anchors (point-to-plane, robust) to pin absolute Z + tilt and pull
  the scan's systematic bias out using the trusted source. This is the
  heterogeneous-source fusion -- probe and scan covering each other's failure
  modes in one mesh.

Hierarchy for "n passes -> decent mesh": pool under the kinematic prior -> robust
surface fit with confidence weights -> constrained ICP/BA to remove residual
servo/laser calibration error -> ICP-anchor onto probe points for absolute truth.
If cooperative passes agree under the prior, ICP for *merging* may be skippable
entirely; its real jobs are residual self-calibration and probe-anchoring.

## Graceful degradation

Optically uncooperative regions (clear acrylic, polished metal, wet/oily wood)
come back low-confidence and **fall back to a touch probe in just those spots**,
while cooperative regions get the dense scan. Because every sample carries
`source` + confidence and the mesh is a point sink, a per-region *mix* of scanned
and probed points fuses into one surface with no special-casing -- the reason the
frame is source-tagged in the first place.

## Build order (when the side project starts)

1. Fixed line laser + existing camera, manual single profile -> validate the
   triangulation math and the camera intrinsics/extrinsics calibration.
2. Machine-swept single-azimuth scan -> point cloud in machine coords (step-and-
   shoot), feeding `ZSurface.samples`.
3. Add the yaw servo + per-angle laser-plane calibration -> multi-azimuth.
4. Confidence model + robust fusion; then constrained-ICP self-calibration.
5. Process isolation + the `Source`/IPC contract; adaptive azimuths;
   scan->probe anchoring.
