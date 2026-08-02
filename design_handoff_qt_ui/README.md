# Handoff: realWorldGcodeSender — Qt UI

## Overview
A vision-guided CNC G-code sender. The app takes a photo of the CNC bed (with the
workpiece on it), detects ChArUco markers on the left/right rails, projects the
image to an overhead bed view, and overlays where the loaded G-code / SVG will
cut. From there the user places & rotates the toolpath, probes the workpiece
(touch-off, Z-mesh, edge-refine), and streams the job to a grbl controller —
watching a live router-position marker on the bed image as it cuts.

This package specifies a **Qt desktop UI** to replace the current
matplotlib-window + tkinter-config-dialog interface.

## About the design files
The files in `screens/` are **design references authored in HTML** (a design-system
called "Classical"). They are prototypes of the intended look, layout, and states —
**not production code to copy**. The task is to **recreate them as a Qt UI inside the
existing Python codebase** (`realWorldGcodeSender/`), using that project's patterns.

Recommended stack: **PySide6** (Qt 6). The repo already anticipates this — see
`ui/interfaces/base.py` (`UIInterface`) and `ui/interfaces/matplotlib_ui.py`
(`MatplotlibUI`). Implement a sibling `QtUI(UIInterface)` and wire it to the
existing startup pipeline; do **not** port the whole modular `machine/` tree
(PROBING_DESIGN.md is explicit that the running monolith `realWorldGcodeSender.py`
is the real codebase).

## Fidelity
- **High-fidelity** — the five app screens and three probing storyboards. Final
  colors, type, spacing, iconography, and states. Recreate pixel-accurately using
  the tokens in this doc.
- **Low-fidelity** — `screens/QtUI Wireframes.dc.html` (layout exploration only).
  Use for structure/flow reference; the hi-fi screens are the source of truth for
  styling.

---

## Target architecture (how it plugs into the existing code)

| UI concept | Qt construct | Backed by (existing code) |
|---|---|---|
| Window shell | `QMainWindow` | new `QtUI(UIInterface)` in `ui/interfaces/qt_ui.py` |
| Menu / toolbar | `QMenuBar`, `QToolBar` | actions call `GCodeSender` methods |
| Workspace tabs | `QTabWidget` (or custom top strip) | one view per tab |
| Left / right docks | `QDockWidget` | — |
| Bed view (center) | `QGraphicsView` + `QGraphicsScene` | overhead pixmap from `warp_to_overhead()`; overlay = scene items |
| G-code/probe/router overlays | `QGraphicsPathItem` / `QGraphicsItem` | path points, `propose_z_targets`, live `WPos` |
| DRO / console / status | `QLabel`s, `QPlainTextEdit`, `QStatusBar` | `GCodeSender.get_absolute_pos()`, grbl responses |
| Settings form | `QLineEdit`/`QCheckBox` in a scroll area | `app_config.py` (`Configuration`, `get_config`, `save_config`) |

**Pipeline (unchanged, in `realWorldGcodeSender.py`):**
`capture_bed_image()` → `calibrate_bed()` → `warp_to_overhead()` →
`OverlayGcode` / `locate_touch_plate()`. The Qt view consumes the warped frame
as a `QImage`/`QPixmap` instead of handing it to matplotlib.

**Backend seams to bind to:**
- Machine control: `GCodeSender` (`home_machine`, `absolute_move`,
  `work_offset_move`, `set_work_coord_offset`, `set_inches`/`set_mm`,
  `gerbil.hold/resume/killalarm`), and `GCodeSenderMachine` (probe protocol).
- Probing: `probing/strategies.py` via `get_strategy("z_mesh"|"z_touch_off"|
  "edge_refine")`, `probing/targets.py` (`propose_z_targets`,
  `propose_edge_targets`), `workpiece_frame.py` (`WorkpieceFrame`, `Measured`,
  `Source`, `ZSurface`).
- Send: `send_svf`, `send_file`, `send_drawnPoints` (warp via
  `toolpath_warp.warp_gcode_lines` when the frame has a mesh).
- Config: `app_config.py`.
- Long ops run off the UI thread — mirror `GCodeSender.run_async` with a
  `QThread`/worker + signals so the UI never blocks; recovery keys stay live.

---

## Design tokens (Classical)

Source of truth: `classical-tokens.css` (copied here). Map these to a **Qt Style
Sheet (QSS)** + a `QPalette`. Never hard-code values the tokens carry.

**Color**
| Role | Hex | Use |
|---|---|---|
| bg | `#f3f2f2` | window / page ground |
| surface | `#eae9e9` | title bar, plate mat |
| text | `#201f1d` | primary text |
| accent (base) | `#b68235` | strokes, active states, live marker |
| accent-400 | `#e1ad66` | large stage numerals |
| accent-700 | `#7d5411` | DRO figures, overlay strokes |
| accent-800 | `#5a3b0a` | text on accent tints |
| accent-100 | `#fff3e4` | `tag-accent` fill |
| divider | `rgba(32,31,29,.16)` | hairline rules/borders |
| neutral-100 | `#f8f4f4` | dock backgrounds |
| neutral-200 | `#eae7e7` | bed-view canvas ground |
| neutral-400 | `#bab6b6` | window border, disabled dots |
| neutral-600 | `#7d7979` | secondary labels |
| neutral-700 | `#605d5d` | console text, body-muted |
| neutral-800 | `#444141` | menu items |

**Added alert role (not in Classical — introduced for machine-error states):** a
restrained brick red, applied as stroke/tint only (matches the kill-alarm accent).
- alert stroke `#a3352a`, alert text `#7a2018`. Use for feed-hold/kill/alarm,
  failed detection, disabled-critical. Keep it hairline; never a filled block.

**Typography**
- Headings / figures: **Cormorant Garamond** (`--font-heading`), weight 600 max
  (interface); display sizes take normal weight. Numeric readouts use
  `font-feature-settings:"tnum"` (tabular).
- Body: **Lora** (`--font-body`), 15px/1.55.
- Console/G-code only: a monospace face (tabular) — the one functional exception
  to the serif system.
- Scale used: h2 38px, h4 20–22px, h6 13px (uppercase, .08em tracking), body 13–15px,
  labels 11–12px, DRO figures 27–29px, stage numerals 34px.

**Spacing / radius / shadow**
- Spacing scale (px): 4.6 / 9.2 / 13.8 / 18.4 / 27.6 / 36.8 (density 1.15×).
- Radius: sm 2px, md 4px, lg 7px.
- Shadow: sm `0 1px 2px rgba(45,43,43,.14)`, md `0 3px 10px …/.16`,
  lg `0 12px 32px …/.22`. Elevation is a whisper — don't over-shadow.

**Component conventions (from the system)**
- Buttons are **outlined**, never filled. Primary = 1px accent border on
  transparent; hover = 12% accent tint; secondary = divider border.
- Cards/panels are bordered, unfilled. Structure carried by hairline rules.
- Photographs go through the `.plate` treatment: 6px surface-color mat + 1px
  divider outline + a warm archival grade (`filter: sepia(.22) saturate(.82)
  contrast(1.05)`). In Qt, apply the mat as a frame; the sepia grade can be baked
  into the pixmap or skipped for live video.
- Focus: 2px accent ring, 2px offset. `::selection` = 30% accent tint.
- Tags encode fidelity: `tag-accent` (filled gold) = probe / highest;
  `tag-outline` = manual; `tag-neutral` = vision. Preserve this mapping.

**Icons**
Lucide (https://lucide.dev), 24×24 viewBox, 1.8px stroke, `currentColor`,
rendered at 16px. Bundle the SVGs as Qt resources (`QSvgRenderer`/`QIcon`).
Icons used: folder-open, file-plus (import SVG), camera, video, target/crosshair
(probe), grid (Z-mesh), scan (edge-refine), play, pause, alert-triangle (kill),
home, rotate-ccw (recapture/reset), lock (unlock), move (jog), sliders,
chevrons/arrows (jog directions), settings (cog), layers/panels (workspace),
save, eye (vision), monitor (comm), check.

---

## Global shell (every app screen)

Fixed reference size **1300 × 824** (design canvas; the real window is resizable —
docks/central widget flex). Top-to-bottom:

1. **Title bar** (h40, surface bg, bottom hairline): 3 neutral-400 dots (window
   controls — Qt provides these natively), brand "realWorldGcodeSender" (Cormorant
   600, 16px), italic screen name in neutral-600, right-aligned uppercase context
   label (neutral-600, 12px, .06em).
2. **Menu bar** (h32): File · Edit · View · Machine · Probe · Help (13.5px, neutral-800).
3. **Workspace tabs** (h40, bottom hairline): Calibration · Workspace · Machine ·
   Settings. Underline tabs — inactive neutral-600, hover→text + neutral-400
   underline, **active = text + 2px accent underline**. Each has a 16px Lucide icon.
4. **Toolbar** (neutral-100 bg): grouped outlined buttons separated by 1px×24 dividers.
5. **Body**: docks + central view (per screen, below).
6. **Console** (h~88–96, neutral-100): uppercase "Console" label + right meta;
   monospace log lines, accent `>` prompt, a blinking caret.
7. **Status bar** (h34, surface): live dot + "Connected · COM3", units, job/step,
   right-aligned uppercase state (IDLE/RUN/ERROR).

---

## Screens

### 1. Workspace  (`screens/QtUI Main - Classical.dc.html`) — hifi, default tab
**Purpose:** place the toolpath on the bed photo, probe, and send.
- **Toolbar:** Open · Import SVG | Recapture · [Still|Live] segmented | Track head
  (primary, active) | Touch-off Z · Z-Mesh · Edge refine | Send job (primary) ·
  Hold · Resume · Kill alarm · Home.
- **Left dock (290px, neutral-100):** *Job & Paths* — [SVG|G-code] segmented,
  source-file field; *Cut parameters* — 2-col grid (thickness, cutter ⌀, feed,
  depth/pass, safe height, tab height); *Paths* table (18 rows, "All paths"
  highlighted with `tag-accent`), helper note "Click the bed to place · right-drag
  to rotate · n/p select path".
- **Center (neutral-200):** kicker "Still capture · cnc13.jpg · overhead-warped";
  right of it a **live Router readout** (blinking dot + `4.81, 7.33 in`) and the
  clicked-point coordinate. The bed image (`ref/cutPath.png`) in a `.plate` mat,
  `object-fit:contain`. **Live router marker** overlaid at its XY: a pulsing
  accent ring (`routerPulse`, 1.9s), a gold crosshair (accent-700), and a
  coordinate tag with a blinking live dot.
- **Right dock (322px):** *Position* — header carries a blinking live dot + "live ·
  WCS · in"; large accent-700 tabular DRO (X/Y/Z, 29px). *Workpiece frame* — table:
  X/Y origin→`vision` (neutral tag), angle→`manual` (outline tag), Z surface→`probe
  · mesh` (accent tag); note "Vision proposes, probe disposes". *Probe* — 2×2
  outlined buttons (Touch-off Z, Z-Mesh, Edge refine, XYZ plate). *Jog* — 3×3 XY
  pad with center home + Z± column.
- **Status bar:** Job `458 / 1,204` with a 150px progress bar; state RUN (if a job
  is running) else IDLE.

### 2. Calibration  (`screens/QtUI Calibration - Classical.dc.html`) — hifi
**Purpose:** detect rail markers, enter physical measurements, confirm homography.
- **Toolbar:** Camera device dropdown · Recapture · [Still|Live] | Detect markers
  (primary) · Camera calibration… | Reset.
- **Center:** the **raw angled** camera photo (`ref/cnc13.jpg`) in a 4:3 `.plate`
  box (`object-fit:cover` so the overlay registers exactly). Gold detection
  overlay: bed-plane quad (accent-700 dashed, accent 10% fill), left & right rail
  boxes, numbered marker chips (left 33–36, right 0–3, dark `#7d5411` squares white
  numerals), touch-plate id 66 flagged. **Overlay coords must be computed from the
  real homography at runtime — the mock positions are illustrative.**
- **Right dock (340px):** *Detection* table (marker counts as accent tags,
  touch-plate found, homography residual `0.6 px`) + a quality meter (~84%);
  *Physical setup* — Bed X/Y/Z + ChArUco box width; *Left/Right reference box* —
  X/Y/Z-offset each.
- **Bottom drawer:** overhead-warp preview (`ref/cutPath.png` plate) + explanation
  + Save to config.json / Accept & open workspace (primary).

### 3. Calibration — error state  (`screens/QtUI Calibration Error - Classical.dc.html`) — hifi
Same screen, **marker-not-found** state (use the alert role):
- Alert banner under the toolbar: alert-triangle + "Left rail markers not detected."
  + cause/fix text + Re-detect button (alert-outlined).
- Overlay: left rail = red dashed search region with hollow `?` marker slots;
  touch-plate "lost"; bed-plane quad faint/dashed ("needs both rails").
- Right dock: Left `0/4` and touch-plate "not found" as brick-tint tags, residual
  "—", quality meter red "failed".
- Bottom: "no overhead view yet" placeholder; **Accept disabled** (45% opacity).
- Status: "Calibration invalid" + state DETECTION FAILED (alert text).

### 4. Machine  (`screens/QtUI Machine Control - Classical.dc.html`) — hifi
**Purpose:** jog & machine state, **with the bed kept in view** (not full-screen).
- **Toolbar:** Open · Recapture | Home ($H) · Unlock ($X) | Send job · Hold ·
  Resume · Kill alarm (alert).
- **Left dock (296px):** *Coordinates* — [in|mm], WCS DRO with per-axis zero
  buttons (X0/Y0/Z0), compact MCS block, Go-to grid (XY zero / Z safe / work zero /
  machine 0), note "press m to move to last clicked bed point"; *Spindle* — [On|Off]
  + RPM.
- **Center:** bed photo with the same live router marker (labelled "head").
- **Right dock (360px):** *Jog* — state tag, Step segmented (.001/.01/.1/1.0),
  3×3 XY pad + Z± column, Feed slider (1,800 ipm); *Overrides* — Feed/Rapid/Spindle
  with −/value%/+ steppers; *Recovery* — Feed hold / Resume / Kill alarm (alert).
- **Console:** real grbl jog lines (`$J=G91 G20 X0.1 F1800`, `<Idle|WPos:…|FS:…>`).

### 5. Settings  (`screens/QtUI Settings - Classical.dc.html`) — hifi
**Purpose:** edit `config.json` (replaces `config_gui.py`).
- **Left category nav (224px):** Physical setup · Cutting parameters · Vision ·
  Communication · Probing (anchor links; active = accent left-border + 10% tint).
- **Form (scroll area):** sections separated by `.hr` hairlines, fields in 2/3-col
  grids — every field from `app_config.py`:
  - Physical: ChArUco box width; Bed X/Y/Z; Right & Left ref box (X, Y, Z-offset,
    far-end height).
  - Cutting: material thickness, cutter diameter, cut feed rate, depth per pass,
    depth below material, safe height, tab height.
  - Vision: bed view size (px), camera device index, camera width, camera height.
  - Communication: COM port, baud rate, auto-detect port (checkbox).
  - Probing: touch-plate height, touch-plate width, distance to notch, probe feed
    fast, probe feed slow.
- **Action bar (h60):** Load default · Load from file… | "Unsaved changes" | Apply ·
  Save as… · Save (primary). Maps to `Configuration.load/save`, `save_config()`.

### 6. Probing storyboards (hifi, reference explainers — not app screens)
`screens/QtUI Probing Flow - Classical.dc.html` (Z-mesh),
`QtUI Edge Refine - Classical.dc.html`, `QtUI Touch-Plate Zeroing - Classical.dc.html`.
Editorial 4-panel spreads documenting each probing flow's stages and the fidelity
each raises. Use them to understand the probing UX and to design in-app progress /
target-review affordances (they are not themselves windows to build).

---

## Interactions & behavior
- **Keyboard (preserve from the monolith):** `s`/`C` send, `g` send file, `n`/`p`
  select path, `h` home, `m` move-to-clicked, `z` XYZ touch-off, `Z` Z-mesh,
  `d`/`c`/`e`/`E` draw, `space` feed-hold, `r` resume, `x` kill-alarm. Recovery
  keys (`space`/`r`/`x`) are **never** disabled while busy.
- **Bed interactions:** left-click places the toolpath (sets X/Y offset);
  right-drag rotates; clicking shows the point's inch coordinate.
- **Live updates (poll grbl status ~5–10 Hz):** DRO figures, router marker XY,
  job progress, state word, override percentages. Bind marker position to `WPos`.
- **Camera:** Still (one capture) vs Live (video frames into the same view/warp).
- **States to build:** connected/disconnected, idle/run/hold/alarm, job
  running/paused/done, calibration valid/invalid (error screen), probe
  running/failed, unsaved-settings.
- **Animations:** router pulse ring 1.9s ease-out infinite; live dots blink 1.4s.
  Keep motion subtle and only for genuinely live values.

## State management
- `WorkpieceFrame` (XY origin, angle, Z surface) — each value source-tagged
  (`vision`/`manual`/`probe`); drives the frame table & fidelity tags.
- Machine: WCS + MCS position, state, feed/rapid/spindle overrides, homed/alarm/
  limits flags, connection.
- Job: current line / total, % , elapsed/remaining, current path, feed/spindle.
- Config: the `Configuration` object; dirty flag for "unsaved changes".

## Assets
- `ref/cnc13.jpg` — raw angled bed photo (4096×3072). Used on the Calibration
  screen; at runtime this is the live/last camera frame.
- `ref/cutPath.png` — overhead-warped bed with G-code overlay (868×897). Used on
  Workspace/Machine and the warp preview; at runtime it's `warp_to_overhead()`
  output with the live overlay drawn on top.
- `classical-tokens.css` — the Classical design-system stylesheet (token source).
- Lucide icons — see icon list above; fetch from lucide.dev and bundle as resources.

## Files (in this bundle, under `screens/`)
- `QtUI Main - Classical.dc.html` — Workspace (hifi)
- `QtUI Calibration - Classical.dc.html` — Calibration (hifi)
- `QtUI Calibration Error - Classical.dc.html` — Calibration error state (hifi)
- `QtUI Machine Control - Classical.dc.html` — Machine / jog (hifi)
- `QtUI Settings - Classical.dc.html` — Settings (hifi)
- `QtUI Probing Flow - Classical.dc.html` — Z-mesh storyboard (hifi)
- `QtUI Edge Refine - Classical.dc.html` — edge-refine storyboard (hifi)
- `QtUI Touch-Plate Zeroing - Classical.dc.html` — touch-plate storyboard (hifi)
- `QtUI Wireframes.dc.html` — layout explorations (lofi)
- `ref/` — image assets · `classical-tokens.css` — design tokens
- `screenshots/` — full-window renders of each hi-fi screen (01 Workspace,
  02 Calibration, 03 Calibration error, 04 Machine, 05 Settings, 06–08 probing
  storyboards) at the 1300-wide reference size, for visual reference.
- `pyside6_skeleton/` — a runnable PySide6 scaffold of the app shell (tabs, docks,
  panels, `classical.qss` theme, and a `QtUI(UIInterface)` adapter stub). See its
  own README to run and for the wiring checklist.

> The `.dc.html` files are streaming design-component prototypes; they render in the
> authoring tool, not standalone. This README is self-sufficient — implement from it,
> using the HTML only as a visual reference.
