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

## Explicitly out of scope

The broader `REFACTORING_PLAN.md` (Qt/Web UI, 3D, protobuf calibration, dynamic
plugin manager, etc.). Not pursued here; the modular tree is left in place but
not extended.
