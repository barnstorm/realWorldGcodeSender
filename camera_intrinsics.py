"""Per-camera lens intrinsics: calibration target detection, storage, undistortion.

Profiles live in camera_intrinsics.json keyed by camera device index, each
holding the camera matrix, distortion coefficients and the frame size they
were computed at.  capture_bed_image() calls undistort_frame() on every live
frame, so the rest of the vision pipeline (homographies, overlays) sees a
distortion-free image; when no profile exists the frame passes through
untouched.

Two calibration targets are supported, tried in this order:
  1. The LightBurn AprilTag sheet (CalibrationTags.pdf): a 6x9 grid of
     36h11 tags, IDs 42-95, 30 mm pitch, 20 mm squares.  Any subset of
     visible tags works, so partial views and odd angles are fine.
  2. A 9x6 inner-corner checkerboard (page 4 of markers/marker_sheets.pdf),
     which must be fully visible.
Intrinsics are scale-free, so print size does not matter -- only the grid
geometry ratios do.
"""

import json
from pathlib import Path

import cv2
import numpy as np

STORE_PATH = Path(__file__).resolve().parent / "camera_intrinsics.json"
CHECKERBOARD_SIZE = (9, 6)     # inner corners (columns, rows)
MIN_CAPTURES = 8

# LightBurn CalibrationTags.pdf layout, measured from the PDF at 300 dpi:
# ID 42 bottom-left, ascending left-to-right then upward, in a y-down frame.
APRILTAG_FIRST_ID = 42
APRILTAG_GRID = (6, 9)         # columns, rows
APRILTAG_PITCH_MM = 30.0
APRILTAG_SIDE_MM = 20.0
MIN_TAGS = 6

_profiles = None
_undistort_maps = {}


def _load_store():
    global _profiles
    if _profiles is None:
        _profiles = (json.loads(STORE_PATH.read_text(encoding="utf-8"))
                     if STORE_PATH.exists() else {})
    return _profiles


def load_profile(device_index):
    return _load_store().get(str(device_index))


def save_profile(device_index, camera_matrix, dist_coeffs, image_size, rms,
                 camera_name=""):
    global _profiles
    profiles = dict(_load_store())
    profiles[str(device_index)] = {
        "camera_name": camera_name,
        "camera_matrix": np.asarray(camera_matrix).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs).ravel().tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "rms": float(rms),
    }
    STORE_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    _profiles = None
    _undistort_maps.clear()


def _to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


# One place for the OpenCV 4/5 aruco API differences; every detection in the
# project goes through these.
_dictionaries = {}
_detectors = {}


def aruco_dictionary(dict_id):
    if dict_id not in _dictionaries:
        _dictionaries[dict_id] = (cv2.aruco.getPredefinedDictionary(dict_id)
                                  if hasattr(cv2.aruco, "getPredefinedDictionary")
                                  else cv2.aruco.Dictionary_get(dict_id))
    return _dictionaries[dict_id]


def detect_aruco(image, dict_id):
    """detectMarkers on a BGR or gray image; returns (boxes, ids)."""
    gray = _to_gray(image)
    if hasattr(cv2.aruco, "detectMarkers"):
        boxes, ids, _rejected = cv2.aruco.detectMarkers(gray, aruco_dictionary(dict_id))
    else:
        if dict_id not in _detectors:
            _detectors[dict_id] = cv2.aruco.ArucoDetector(aruco_dictionary(dict_id))
        boxes, ids, _rejected = _detectors[dict_id].detectMarkers(gray)
    return boxes, ids


def tag_object_corners(tag_id):
    """Sheet-frame corners (mm, y-down, z=0) of one tag, in OpenCV's detected
    order for these tags: bottom-right, bottom-left, top-left, top-right
    (AprilTag corner convention, measured from the rendered PDF)."""
    n = tag_id - APRILTAG_FIRST_ID
    cols, rows = APRILTAG_GRID
    col, row = n % cols, n // cols
    cx = col * APRILTAG_PITCH_MM
    cy = (rows - 1 - row) * APRILTAG_PITCH_MM
    h = APRILTAG_SIDE_MM / 2.0
    return np.array([[cx + h, cy + h, 0], [cx - h, cy + h, 0],
                     [cx - h, cy - h, 0], [cx + h, cy - h, 0]], np.float32)


def find_tags(frame):
    """Detect LightBurn sheet tags; any MIN_TAGS+ subset is a usable view."""
    boxes, ids = detect_aruco(frame, cv2.aruco.DICT_APRILTAG_36h11)
    if ids is None:
        return None
    cols, rows = APRILTAG_GRID
    last_id = APRILTAG_FIRST_ID + cols * rows - 1
    hits = [(box, int(i)) for box, i in zip(boxes, ids.flatten())
            if APRILTAG_FIRST_ID <= int(i) <= last_id]
    if len(hits) < MIN_TAGS:
        return None
    image_points = np.concatenate([box.reshape(4, 1, 2) for box, _ in hits])
    object_points = np.concatenate([tag_object_corners(i) for _, i in hits])
    return {"kind": "apriltag", "count": len(hits),
            "image_points": image_points.astype(np.float32),
            "object_points": object_points,
            "outlines": [box.reshape(4, 2) for box, _ in hits]}


def find_checkerboard(frame, fast=False):
    gray = _to_gray(frame)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, flags=flags)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    cols, rows = CHECKERBOARD_SIZE
    object_points = np.zeros((cols * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return {"kind": "checkerboard", "count": len(corners),
            "image_points": corners, "object_points": object_points,
            "corners": corners}


def find_target(frame, fast=False):
    """Detect whichever calibration target is in view (AprilTags first)."""
    return find_tags(frame) or find_checkerboard(frame, fast)


def calibrate(views, image_size):
    """Compute intrinsics from detected views (dicts from find_target).

    Returns (rms, camera_matrix, dist_coeffs).  image_size is (width, height).
    """
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        [view["object_points"] for view in views],
        [view["image_points"] for view in views],
        image_size, None, None)
    return rms, camera_matrix, dist_coeffs


def rotate_camera_matrix(camera_matrix, rotation, size):
    """Pinhole matrix after cv2.rotate of the image.  size is (w, h) before
    rotation; rotation is degrees clockwise in {0, 90, 180, 270}."""
    w, h = size
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if rotation == 90:      # (x, y) -> (h-1-y, x)
        fx, fy, cx, cy = fy, fx, h - 1 - cy, cx
    elif rotation == 180:   # (x, y) -> (w-1-x, h-1-y)
        cx, cy = w - 1 - cx, h - 1 - cy
    elif rotation == 270:   # (x, y) -> (y, w-1-x)
        fx, fy, cx, cy = fy, fx, cy, w - 1 - cx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def _profile_matrices(device_index, size):
    """(camera_matrix, dist_coeffs, new_matrix) for pre-rotation frames of
    (width, height), or None without a matching profile.  alpha=1 keeps the
    full field of view (black curved borders instead of cropping) so markers
    near the frame edges survive -- undistortion and scene_camera_matrix must
    share this choice, which is why both go through here."""
    profile = load_profile(device_index)
    if profile is None or list(size) != profile["image_size"]:
        return None
    camera_matrix = np.array(profile["camera_matrix"])
    dist_coeffs = np.array(profile["dist_coeffs"])
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, tuple(size), 1, tuple(size))
    return camera_matrix, dist_coeffs, new_matrix


def scene_camera_matrix(device_index, rotation, scene_size):
    """Effective pinhole matrix of frames leaving to_scene_frame(): the
    undistorted image's optimal new matrix, then the rotation applied.
    scene_size is the delivered frame's (width, height) AFTER rotation.
    Returns None when there is no profile or the sizes do not match (in which
    case the frame was not undistorted either)."""
    profile = load_profile(device_index)
    if profile is None:
        return None
    rotation = rotation % 360
    w, h = profile["image_size"]
    expected = (h, w) if rotation in (90, 270) else (w, h)
    if tuple(scene_size) != expected:
        return None
    matrices = _profile_matrices(device_index, (w, h))
    return rotate_camera_matrix(matrices[2], rotation, (w, h))


def undistort_frame(frame, device_index):
    """Undistort using the stored profile; pass through when none applies."""
    height, width = frame.shape[:2]
    key = str(device_index)
    cached = _undistort_maps.get(key)
    if cached is None or cached[0] != (width, height):
        matrices = _profile_matrices(device_index, (width, height))
        if matrices is None:
            return frame
        camera_matrix, dist_coeffs, new_matrix = matrices
        maps = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_matrix,
            (width, height), cv2.CV_16SC2)
        cached = _undistort_maps[key] = ((width, height), maps)
    return cv2.remap(frame, cached[1][0], cached[1][1], cv2.INTER_LINEAR)


def to_scene_frame(frame, device_index, rotation):
    """Undistort in sensor orientation, then rotate to scene orientation --
    the pixel-side twin of scene_camera_matrix()."""
    frame = undistort_frame(frame, device_index)
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotation % 360)
    return frame if code is None else cv2.rotate(frame, code)
