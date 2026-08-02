"""Generate printable sheets for the rail ArUco strips and touch-plate marker.

The strip geometry comes straight from idToLocDict and box_width, so the
printout matches what calibrate_bed() expects: every marker is box_width
square with its corners on the 3x22 cell grid.  Marker rotation on the sheet
does not matter (sortBoxPoints orders corners geometrically).

Usage:
    python generate_markers.py            # writes markers/marker_sheets.pdf + PNGs
    python generate_markers.py --dpi 600

Print the PDF at 100% scale / "actual size" and check the ruler on each page
before cutting.  Tape the two segments of each rail together at the dashed
joint line, ID-0/33 end toward the machine rear (Y origin side).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app_config import get_config
from camera_intrinsics import aruco_dictionary
from realWorldGcodeSender import idToLocDict

PAGE_W, PAGE_H = 8.5, 11.0
MARGIN = 0.5
RIGHT_RAIL_IDS = [i for i in idToLocDict if i <= 32]
LEFT_RAIL_IDS = [i for i in idToLocDict if i >= 33]
TOUCH_PLATE_ID = 66


def marker_image(dictionary, marker_id, side_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_px)


def strip_segment(ids, row_lo, row_hi, cell_px):
    """Render the markers of one rail whose grid row is in [row_lo, row_hi].

    Row row_hi renders at the top, column 2 at the right -- the "first ID is
    upper right" orientation from the idToLocDict comment.  Returns the image
    and its size in cells (cols, rows).
    """
    cols = 3
    rows = row_hi - row_lo + 1
    img = np.full((rows * cell_px, cols * cell_px), 255, np.uint8)
    dictionary = aruco_dictionary(cv2.aruco.DICT_4X4_100)
    for marker_id in ids:
        col, row = idToLocDict[marker_id]
        if not row_lo <= row <= row_hi:
            continue
        y0 = (row_hi - row) * cell_px
        x0 = col * cell_px
        img[y0:y0 + cell_px, x0:x0 + cell_px] = \
            marker_image(dictionary, marker_id, cell_px)
    return img, cols, rows


def draw_ruler(ax, x, y, inches=5):
    ax.plot([x, x + inches], [y, y], color="black", lw=1)
    for tick in range(inches + 1):
        ax.plot([x + tick, x + tick], [y, y + 0.12], color="black", lw=1)
        ax.text(x + tick, y + 0.18, str(tick), ha="center", va="bottom", fontsize=7)
    ax.text(x + inches / 2.0, y - 0.1,
            "scale check: ticks are 1 inch apart -- print at 100%",
            ha="center", va="top", fontsize=7)


def new_page(pdf, title):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(MARGIN, PAGE_H - MARGIN + 0.15, title, fontsize=11, va="bottom")
    draw_ruler(ax, MARGIN, MARGIN + 0.3)
    return fig, ax


def place_segment(ax, img, x, y, box_width, cols, rows, label):
    w, h = cols * box_width, rows * box_width
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x, x + w, y, y + h), zorder=2)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                               edgecolor="0.6", lw=0.5, zorder=3))
    ax.text(x + w / 2.0, y - 0.08, label, ha="center", va="top", fontsize=8)
    return w, h


def rail_page(pdf, name, ids, box_width, cell_px):
    """One rail per page: rows 11-21 segment beside rows 0-10 segment."""
    fig, ax = new_page(pdf, "%s rail markers (IDs %d-%d), box width %.6f in"
                       % (name, min(ids), max(ids), box_width))
    top_img, cols, top_rows = strip_segment(ids, 11, 21, cell_px)
    bot_img, _, bot_rows = strip_segment(ids, 0, 10, cell_px)
    x = MARGIN + 0.6
    y = PAGE_H - MARGIN - 0.6 - top_rows * box_width
    w, h = place_segment(ax, top_img, x, y, box_width, cols, top_rows,
                         "rows 11-21  (this end toward machine rear / Y origin)")
    x2 = x + w + 1.2
    place_segment(ax, bot_img, x2, y + h - bot_rows * box_width, box_width,
                  cols, bot_rows, "rows 0-10  (front end)")
    for seg_x, seg_y in ((x, y), (x2, y + h - bot_rows * box_width)):
        ax.plot([seg_x - 0.15, seg_x + cols * box_width + 0.15],
                [seg_y, seg_y] if seg_x == x else
                [seg_y + bot_rows * box_width] * 2,
                color="0.4", lw=0.8, ls="--", zorder=1)
    ax.text(x + w / 2.0, y - 0.35,
            "cut both segments at the dashed line and butt them together:\n"
            "row 10 (top of right segment) sits directly below row 11",
            ha="center", va="top", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def checkerboard_page(pdf, dpi, square_in=0.9):
    """Lens-calibration checkerboard: 9x6 inner corners (10x7 squares).

    The square size does not affect the computed intrinsics -- only the
    inner-corner count matters (camera_intrinsics.CHECKERBOARD_SIZE).
    """
    squares_x, squares_y = 7, 10
    fig, ax = new_page(pdf, "Lens calibration checkerboard "
                            "(9x6 inner corners, %.1f in squares)" % square_in)
    cell = round(square_in * dpi)
    img = np.full((squares_y * cell, squares_x * cell), 255, np.uint8)
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                img[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell] = 0
    w, h = squares_x * square_in, squares_y * square_in
    x = PAGE_W / 2.0 - w / 2.0
    y = PAGE_H / 2.0 - h / 2.0 + 0.4
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
              extent=(x, x + w, y, y + h), zorder=2)
    ax.text(x + w / 2.0, y - 0.15,
            "mount on something rigid (cardboard/clipboard) and keep it flat",
            ha="center", va="top", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def machine_strip_pages(pdf, dpi, config):
    """Flat AprilTag strips for the rail tops and bed edges (machine_tags.py).

    Each page holds one segment of the right and left strip of a pair.  Cut
    the columns apart and mount each tag with its center tick at the labeled
    machine Y, along the labeled X centerline.  Tag rotation does not matter
    (calibration uses tag centers), only position does.
    """
    import machine_tags as mt
    p = config.physical_setup
    size, pitch = p.machine_tag_size, p.machine_tag_pitch
    count = mt.strip_count(config)
    tag_px = round(size * dpi)
    dictionary = aruco_dictionary(cv2.aruco.DICT_APRILTAG_36h11)
    label = lambda name, x_attr, on_rail: "%s%s  X %+.2f" % (
        name.upper(), " TOP" if on_rail else " EDGE", getattr(p, x_attr))
    groups = tuple(
        tuple((label(name, x_attr, on_rail), first_id)
              for first_id, name, x_attr, on_rail in mt.STRIPS if on_rail == rail)
        for rail in (True, False))
    per_page = 3
    segments = (count + per_page - 1) // per_page
    for pair in groups:
        for seg in range(segments):
            fig, ax = new_page(pdf, "Machine tag strips, segment %d/%d - "
                                    "%.2f in tags every %.2f in"
                               % (seg + 1, segments, size, pitch))
            for column, (name, first_id) in enumerate(pair):
                x_center = 2.1 + column * 4.2
                top_y = PAGE_H - MARGIN - 1.1
                ax.text(x_center, top_y + 0.45, name, ha="center", fontsize=9)
                ax.annotate("", xy=(x_center - size / 2 - 0.45, top_y + 0.55),
                            xytext=(x_center - size / 2 - 0.45, top_y + 0.05),
                            arrowprops=dict(arrowstyle="->", lw=1))
                ax.text(x_center - size / 2 - 0.55, top_y + 0.3, "machine REAR (Y=0)",
                        rotation=90, ha="right", va="center", fontsize=6)
                for k in range(seg * per_page, min((seg + 1) * per_page, count)):
                    cy = top_y - size / 2.0 - (k - seg * per_page) * pitch
                    tag_id = first_id + k
                    img = marker_image(dictionary, tag_id, tag_px)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
                              interpolation="nearest", zorder=2,
                              extent=(x_center - size / 2, x_center + size / 2,
                                      cy - size / 2, cy + size / 2))
                    for tick_x in (x_center - size / 2 - 0.4, x_center + size / 2 + 0.1):
                        ax.plot([tick_x, tick_x + 0.3], [cy, cy], color="black", lw=0.8)
                    ax.text(x_center + size / 2 + 0.45, cy,
                            "ID %d\nY %.2f in" % (tag_id, p.machine_tag_y_start - k * pitch),
                            va="center", fontsize=7)
            ax.text(PAGE_W / 2.0, MARGIN + 0.85,
                    "cut columns apart; mount with each center tick at the labeled "
                    "machine Y,\nstrip centered on the labeled X; tag rotation does not matter",
                    ha="center", va="bottom", fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)


def touch_plate_page(pdf, box_width, cell_px):
    fig, ax = new_page(pdf, "Touch plate marker (ID %d), %.6f in square"
                       % (TOUCH_PLATE_ID, box_width))
    img = marker_image(aruco_dictionary(cv2.aruco.DICT_4X4_100), TOUCH_PLATE_ID, cell_px)
    x = PAGE_W / 2.0 - box_width / 2.0
    y = PAGE_H / 2.0
    place_segment(ax, img, x, y, box_width, 1, 1,
                  "glue to the touch plate, any rotation")
    pdf.savefig(fig)
    plt.close(fig)


def save_full_strips(out_dir, box_width, cell_px, dpi):
    from PIL import Image
    for name, ids in (("right_rail", RIGHT_RAIL_IDS), ("left_rail", LEFT_RAIL_IDS)):
        img, _, _ = strip_segment(ids, 0, 21, cell_px)
        path = out_dir / (name + ".png")
        Image.fromarray(img).save(path, dpi=(dpi, dpi))
        print("wrote %s (%.2f x %.2f in at %d dpi)"
              % (path, img.shape[1] / dpi, img.shape[0] / dpi, dpi))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", default="markers")
    args = parser.parse_args()

    box_width = get_config().physical_setup.box_width
    cell_px = round(box_width * args.dpi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "marker_sheets.pdf"
    with PdfPages(pdf_path) as pdf:
        machine_strip_pages(pdf, args.dpi, get_config())
        touch_plate_page(pdf, box_width, cell_px)
        checkerboard_page(pdf, args.dpi)
        rail_page(pdf, "Right", RIGHT_RAIL_IDS, box_width, cell_px)
        rail_page(pdf, "Left", LEFT_RAIL_IDS, box_width, cell_px)
    print("wrote %s" % pdf_path)
    save_full_strips(out_dir, box_width, cell_px, args.dpi)


if __name__ == "__main__":
    main()
