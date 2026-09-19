import math
import numpy as np


def grid_lines(left, right, bottom, top, step=1.0):
    # (point, direction) specs for the unit gridlines inside [left, right] x
    # [bottom, top], skipping x=0/y=0 (those are drawn as axes instead)
    lines = []

    x = math.ceil(left / step) * step
    while x <= right + 1e-9:
        if abs(x) > 1e-9:
            lines.append(((x, 0.0), (0.0, 1.0)))
        x += step

    y = math.ceil(bottom / step) * step
    while y <= top + 1e-9:
        if abs(y) > 1e-9:
            lines.append(((0.0, y), (1.0, 0.0)))
        y += step

    return lines


def fit_bounds(width, height, pixels_per_unit, center=(0.0, 0.0), zoom=1.0):
    # world bounds for a `width` x `height` viewport centred on `center`: one
    # world unit is `pixels_per_unit * zoom` screen pixels on both axes, so
    # resizing changes how much of the world is visible (field of view) while
    # zooming changes how big a world unit looks on screen
    scale = pixels_per_unit * zoom
    half_width = (width / 2.0) / scale
    half_height = (height / 2.0) / scale
    return (center[0] - half_width, center[0] + half_width,
            center[1] - half_height, center[1] + half_height)


def nice_step(world_per_pixel, target_spacing_px=60.0):
    # smallest 1/2/5 x 10^k step whose on-screen spacing reaches
    # target_spacing_px. A step fixed at 1.0 only reads well near zoom 1: zoomed
    # out the gridlines mat into a solid block, zoomed in they leave the screen
    # empty, and either way the grid stops saying anything about scale
    raw = world_per_pixel * target_spacing_px
    if raw <= 0.0:
        return 1.0
    k = math.floor(math.log10(raw))
    mantissa = raw / 10.0 ** k
    for m in (1.0, 2.0, 5.0):
        if mantissa <= m:
            return m * 10.0 ** k
    return 10.0 ** (k + 1)


def axis_lines():
    # the x=0 and y=0 lines, meant to be drawn distinctly from the grid
    return [
        ((0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), (0.0, 1.0)),
    ]


def ortho(left, right, bottom, top, near=-1.0, far=1.0) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m