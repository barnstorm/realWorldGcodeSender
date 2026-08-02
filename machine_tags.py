"""Flat AprilTag strips at known machine coordinates -> camera pose -> bed map.

The strips lie in two horizontal planes -- the rail tops and the bed surface
-- so an overhead-ish camera sees every tag nearly face-on (unlike the legacy
vertical-rail ChArUco strips, which it saw edge-on).  Each tag's center has a
known machine coordinate from the PhysicalSetup config; detecting 6+ of them
plus the lens intrinsics gives the full camera pose via solvePnP, from which
the bed outline is projected and the same bed<->image homographies the rest
of the pipeline consumes are built.

Tag family is AprilTag 36h11.  IDs 0-39 are machine strips (blocks of 10:
right rail top, left rail top, right bed, left bed); IDs 42-95 are the
LightBurn lens-calibration sheet and are ignored here.

Tag centers are computed through each tag's own homography, so strips work
in any mounted rotation -- only the strip's position and direction matter.
"""

import cv2
import numpy as np

import camera_intrinsics

# The single source of truth for the strip scheme: each strip owns a block of
# STRIP_BLOCK consecutive IDs and takes its X centerline from a PhysicalSetup
# attribute.  strip_layout, strip_name, the machine-ID cutoff, and the sheet
# generator all derive from this table.
STRIP_BLOCK = 10
STRIPS = (
    # (first_id, display name, PhysicalSetup X attribute, on rail top)
    (0, "Right rail", "rail_top_right_x", True),
    (10, "Left rail", "rail_top_left_x", True),
    (20, "Right bed", "bed_tags_right_x", False),
    (30, "Left bed", "bed_tags_left_x", False),
)
MACHINE_ID_LIMIT = max(first for first, _n, _a, _r in STRIPS) + STRIP_BLOCK
MIN_TAGS = 6
TOUCH_PLATE_DICT = cv2.aruco.DICT_4X4_100   # legacy plate marker (ID 66)


def strip_count(config):
    """Tags per strip, clamped to the ID block size."""
    return min(int(config.physical_setup.machine_tag_count), STRIP_BLOCK)


def strip_layout(config):
    """{tag_id: (x, y, z) center in machine inches} for all four strips."""
    p = config.physical_setup
    layout = {}
    for first_id, _name, x_attr, on_rail in STRIPS:
        x = getattr(p, x_attr)
        z = p.bed_size_z + (p.rail_top_height if on_rail else 0.0)
        for i in range(strip_count(config)):
            layout[first_id + i] = (x, p.machine_tag_y_start - i * p.machine_tag_pitch, z)
    return layout


def strip_name(tag_id):
    for first_id, name, _x_attr, _on_rail in STRIPS:
        if first_id <= tag_id < first_id + STRIP_BLOCK:
            return name
    return "?"


def _tag_center(box):
    """Projected physical center of a tag -- exact under perspective, and
    invariant to the tag's mounted rotation (centroid of corners is neither)."""
    corners = box.reshape(4, 2).astype(np.float32)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    homography = cv2.getPerspectiveTransform(unit, corners)
    center = cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], np.float32), homography)
    return center[0, 0]


def detect(image):
    """{tag_id: center_px} for every machine tag in view."""
    boxes, ids = camera_intrinsics.detect_aruco(image, cv2.aruco.DICT_APRILTAG_36h11)
    if ids is None:
        return {}, {}
    centers, outlines = {}, {}
    for box, tag_id in zip(boxes, ids.flatten()):
        tag_id = int(tag_id)
        if tag_id < MACHINE_ID_LIMIT:
            centers[tag_id] = _tag_center(box)
            outlines[tag_id] = box.reshape(4, 2)
    return centers, outlines


def solve(centers, layout, camera_matrix):
    """Camera pose from detected tag centers.  Returns (rvec, tvec, rms_px)."""
    ids = sorted(set(centers) & set(layout))
    object_points = np.array([layout[i] for i in ids], np.float64)
    image_points = np.array([centers[i] for i in ids], np.float64).reshape(-1, 1, 2)
    spread = object_points - object_points.mean(axis=0)
    if np.linalg.svd(spread, compute_uv=False)[1] < 1.0:
        raise RuntimeError("machine tags are nearly collinear - at least two "
                           "different strips must be visible")
    # Plain iterative PnP: unlike solvePnPRansac's minimal sampler, it handles
    # the rails-only case (all tags in one plane, on two parallel lines).
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                  None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("camera pose estimation failed on the machine tags")
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points,
                                      camera_matrix, None, rvec, tvec)

    def errors(rv, tv):
        projected, _ = cv2.projectPoints(object_points, rv, tv, camera_matrix, None)
        return np.linalg.norm(projected.reshape(-1, 2) - image_points.reshape(-1, 2),
                              axis=1)

    residuals = errors(rvec, tvec)
    keep = residuals < 8.0   # drop gross misdetections, then re-refine
    if 4 <= keep.sum() < len(keep):
        rvec, tvec = cv2.solvePnPRefineLM(object_points[keep], image_points[keep],
                                          camera_matrix, None, rvec, tvec)
        residuals = errors(rvec, tvec)
        keep = residuals < 8.0
    if keep.sum() < 4:
        raise RuntimeError("machine tag detections are inconsistent - check "
                           "that the strips match the configured positions")
    rms = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    return rvec, tvec, rms


def bed_homographies(rvec, tvec, camera_matrix, config, frame_shape):
    """Project the bed outline and build the legacy bed<->image homographies.

    Matches calibrate_bed's convention exactly: the view's X extent spans the
    RAIL PLANES (right_box_ref_x .. left_box_ref_x) -- the same constants
    BedTransform uses for pixel<->inch mapping -- NOT 0..bed_size_x.  Y spans
    0..bed_size_y.  Corner order is left-back, left-front, right-back,
    right-front against bedPixelCorners [[height,0],[height,width],[0,0],
    [0,width]] (height=shape[1], width=shape[0])."""
    p = config.physical_setup
    z = p.bed_size_z
    x_left = config.get_left_box_ref().X
    x_right = config.get_right_box_ref().X
    machine_corners = np.array([
        [x_left, 0.0, z], [x_left, p.bed_size_y, z],
        [x_right, 0.0, z], [x_right, p.bed_size_y, z]], np.float64)
    ref_pixels, _ = cv2.projectPoints(machine_corners, rvec, tvec, camera_matrix, None)
    ref_pixels = ref_pixels.reshape(-1, 2)
    height, width = float(frame_shape[1]), float(frame_shape[0])
    bed_pixel_corners = np.array([[height, 0.0], [height, width],
                                  [0.0, 0.0], [0.0, width]])
    bed_to_orig, _ = cv2.findHomography(bed_pixel_corners, ref_pixels)
    orig_to_bed, _ = cv2.findHomography(ref_pixels, bed_pixel_corners)
    return bed_to_orig, orig_to_bed, ref_pixels


def _detect_touch_plate(image):
    boxes, ids = camera_intrinsics.detect_aruco(image, TOUCH_PLATE_DICT)
    if ids is None:
        return [], np.zeros((0, 1), np.int32)
    return boxes, np.asarray(ids, np.int32).reshape(-1, 1)


def _annotate(frame, outlines, ref_pixels):
    for tag_id, outline in outlines.items():
        pts = outline.astype(int)
        cv2.polylines(frame, [pts], True, (60, 220, 60), 2)
        cv2.putText(frame, str(tag_id), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    corners = ref_pixels.astype(int)
    for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
        cv2.line(frame, tuple(corners[a]), tuple(corners[b]), (0, 0, 255), 3)


def try_calibrate(frame, config, camera_matrix=None):
    """Full machine-tag calibration; None when no machine tags are in view
    (caller falls back to the legacy vertical-rail path).

    Returns {frame, bed_to_orig, orig_to_bed, boxes, ids, counts, rms}.
    Raises with guidance when tags are visible but unusable.
    """
    layout = strip_layout(config)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    centers, outlines = detect(gray)
    usable = set(centers) & set(layout)
    if len(usable) < MIN_TAGS:
        if usable:
            raise RuntimeError(
                "only %d machine tag%s visible (need %d+) - aim the camera to "
                "see more of the strips" % (len(usable),
                "" if len(usable) == 1 else "s", MIN_TAGS))
        return None
    if camera_matrix is None:
        vision = config.vision_settings
        camera_matrix = camera_intrinsics.scene_camera_matrix(
            vision.camera_device_index, vision.camera_rotation,
            (frame.shape[1], frame.shape[0]))
    if camera_matrix is None:
        raise RuntimeError(
            "machine tags found, but this camera has no lens profile at the "
            "current resolution - run Calibrate lens first")
    rvec, tvec, rms = solve(centers, layout, camera_matrix)
    bed_to_orig, orig_to_bed, ref_pixels = bed_homographies(
        rvec, tvec, camera_matrix, config, frame.shape)
    boxes, ids = _detect_touch_plate(gray)
    _annotate(frame, outlines, ref_pixels)
    counts = {}
    for tag_id in usable:
        name = strip_name(tag_id)
        counts[name] = counts.get(name, 0) + 1
    return {"frame": frame, "bed_to_orig": bed_to_orig, "orig_to_bed": orig_to_bed,
            "boxes": boxes, "ids": ids, "counts": counts, "rms": rms}
