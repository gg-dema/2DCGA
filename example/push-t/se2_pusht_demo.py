"""
Play back Push-T T-block demos in the 2DCGA window.

Each demo is a recorded SE(2) trajectory of the *pushed block* (not the pusher).
Poses are carried as CGA motors: the block outline is a cloud of CGA points that
the motor sandwich transports rigidly, and consecutive samples are bridged by
screw interpolation so playback is smooth at any speed.

By default several demos sharing a start pose are shown at once, so the
multimodality that makes Push-T a benchmark is visible.
"""
import sys
import argparse
import colorsys
from pathlib import Path

import glfw
import numpy as np
from scipy.io import loadmat


sys.path.append(str(Path(__file__).parents[2]))

from src.CGA_2D_op import (even_sandwhich, rotation, rotation_rev,
                           translation, translation_rev)
from src.mv_embedding import point_to_mv, motor_of, extract_pose, gp
from visual.Window import Window
from visual.objects.points import Point
from visual.objects.frames import Frame


DATASET = Path(__file__).resolve().parents[2] / "datasets" / "PushT.cga.mat"

# Push-T uses a 512x512 pixel frame with y pointing DOWN; the window has y UP.
# to_world() negates y (and theta, since mirroring flips the sense of rotation).
GOAL_POSE_PX = (256.0, 256.0, np.pi / 4)   # PushTEnv.goal_pose
FIT_WORLD_UNITS = 8.0                      # world span the scene is scaled to fill

MOVING_FRAME_AXIS_LEN = 0.8                # world units
BREADCRUMB_FRAME_AXIS_LEN = 0.3
BREADCRUMB_MIN_GAP = 0.15                  # world units between path dots
PLAYBACK_SPEED = 1.5                       # 1.0 = recorded wall-clock rate

GOAL_STYLE = {"x": (1.0, 1.0, 1.0, 1.0), "y": (0.60, 0.60, 0.60, 1.0)}

# The T block, as PushTEnv.add_tee builds it: two boxes at scale=30 (a 120x30
# bar and a 30x90 stem), body origin at the bar's bottom centre, in Push-T px.
TEE_SCALE, TEE_LEN = 30.0, 4
TEE_OUTLINE_PX = np.array([
    (-TEE_LEN * TEE_SCALE / 2, 0.0), (TEE_LEN * TEE_SCALE / 2, 0.0),
    (TEE_LEN * TEE_SCALE / 2, TEE_SCALE), (TEE_SCALE / 2, TEE_SCALE),
    (TEE_SCALE / 2, TEE_LEN * TEE_SCALE), (-TEE_SCALE / 2, TEE_LEN * TEE_SCALE),
    (-TEE_SCALE / 2, TEE_SCALE), (-TEE_LEN * TEE_SCALE / 2, TEE_SCALE),
])
OUTLINE_SAMPLE_STEP = 0.07   # world units between outline dots
OUTLINE_DOT_SIZE = 2


def demo_style(index, total):
    """Identity colour per demo: evenly spaced hues, body y darker than body x.

    With a whole group on screen "which demo" reads better than "which axis".
    """
    hue = index / max(total, 1)
    return {
        "dots": colorsys.hsv_to_rgb(hue, 0.80, 1.00) + (1.0,),
        "x": colorsys.hsv_to_rgb(hue, 0.80, 1.00) + (1.0,),
        "y": colorsys.hsv_to_rgb(hue, 1.00, 0.58) + (1.0,),
    }


class StyledRenderer:
    """Bind draw options to a renderer: Window.step calls draw(projection) only."""

    def __init__(self, renderer, **draw_kw):
        self.renderer, self.draw_kw = renderer, draw_kw

    def update(self, *a, **k):
        return self.renderer.update(*a, **k)

    def draw(self, projection):
        self.renderer.draw(projection, **self.draw_kw)


def load_demos(path):
    """demos[0, j] -> {'pos': (3, T), 'dt'}, the LASA layout Squid_v2 writes."""
    data = loadmat(str(path))['demos']
    demos = [data[0, j]['pos'][0, 0].T.astype(float) for j in range(data.shape[1])]
    dt = float(data[0, 0]['dt'][0, 0][0, 0])
    return demos, dt


def find_multimodal_group(demos, k=5, start_caps=(40, 60, 80, 110, 150),
                          rad_to_px=50.0, n_resample=100):
    """k demos that share a start pose but fan out as widely as possible.

    Scored as (mean pairwise path separation) / (spread of their start poses):
    maximising raw separation alone would just drop the shared-start constraint.
    Caps are tried tight -> loose, so the group is as clustered as the data allows.
    """
    # compare at matching phase, which is also how playback runs
    resampled = np.array([
        np.column_stack([np.interp(np.linspace(0, 1, n_resample),
                                   np.linspace(0, 1, len(p)), p[:, c])
                         for c in range(3)])
        for p in demos
    ])
    starts = np.array([p[0] for p in demos])
    # rad_to_px: the block is ~120 px across, so 1 rad of yaw moves it ~50 px
    start_gap = np.hypot(
        np.linalg.norm(starts[:, None, :2] - starts[None, :, :2], axis=-1),
        np.abs(starts[:, None, 2] - starts[None, :, 2]) * rad_to_px,
    )
    np.fill_diagonal(start_gap, 0.0)
    path_sep = np.linalg.norm(
        resampled[:, None, :, :2] - resampled[None, :, :, :2], axis=-1).mean(-1)

    def farthest_first(candidates):
        # seed with the most divergent pair, then keep adding the least similar
        sub = path_sep[np.ix_(candidates, candidates)]
        a, b = np.unravel_index(np.argmax(sub), sub.shape)
        chosen = [candidates[a], candidates[b]]
        while len(chosen) < k:
            rest = [c for c in candidates if c not in chosen]
            if not rest:
                break
            chosen.append(max(rest, key=lambda c: path_sep[c, chosen].min()))
        return chosen

    for cap in start_caps:
        best = None
        for i in range(len(demos)):
            candidates = np.where(start_gap[i] < cap)[0].tolist()
            if len(candidates) < k:
                continue
            group = farthest_first(candidates)
            if len(group) < k:
                continue
            spread = start_gap[np.ix_(group, group)].max()
            if spread > cap:
                continue
            mean_sep = path_sep[np.ix_(group, group)][np.triu_indices(k, 1)].mean()
            score = mean_sep / max(spread, 1.0)
            if best is None or score > best[0]:
                best = (score, sorted(group), spread, mean_sep)
        if best is not None:
            _, group, spread, mean_sep = best
            return group, spread, mean_sep

    raise SystemExit(f"no group of {k} demos shares a start pose within "
                     f"{start_caps[-1]:.0f} px -- ask for fewer")


def place(body_xy, position, theta):
    """Rigidly move body-frame euclidean points to (position, theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return body_xy @ np.array([[c, s], [-s, c]]) + position


def fit_px_per_unit(poses_px, with_block=True):
    """px per world unit making the widest span of the scene fill FIT_WORLD_UNITS.

    The block reaches ~123 px from its origin, comparable to the whole travel of
    that origin, so the outline is swept along each path and folded into the bounds.
    """
    xy = [p[:, :2] for p in poses_px] + [np.array([GOAL_POSE_PX[:2]])]
    if with_block:
        xy.append(place(TEE_OUTLINE_PX, np.array(GOAL_POSE_PX[:2]), GOAL_POSE_PX[2]))
        for p in poses_px:
            idx = np.unique(np.linspace(0, len(p) - 1, 24).round().astype(int))
            xy += [place(TEE_OUTLINE_PX, q[:2], q[2]) for q in p[idx]]
    xy = np.vstack(xy)
    return max((xy.max(0) - xy.min(0)).max(), 1.0) / FIT_WORLD_UNITS


def to_world(poses_px, px_per_unit):
    """(..., 3) Push-T pixel poses -> window world units, goal at the origin."""
    poses_px = np.atleast_2d(poses_px)
    xy = (poses_px[:, :2] - np.array(GOAL_POSE_PX[:2])) / px_per_unit
    return np.column_stack([xy[:, 0], -xy[:, 1], -poses_px[:, 2]])


def body_to_world(body_px, px_per_unit):
    """Body-frame block geometry (px) -> body-frame world units.

    y is mirrored to match to_world's negated theta; without it the T renders
    as its own mirror image.
    """
    return np.column_stack([body_px[:, 0], -body_px[:, 1]]) / px_per_unit


def sample_polygon(polygon, step):
    """Dense samples along a closed polygon: Line draws infinite lines, so the
    block's shape has to be shown as a run of closely spaced dots."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        n = max(int(np.linalg.norm(b - a) / step), 1)
        out.append(a + np.outer(np.linspace(0, 1, n, endpoint=False), b - a))
    return np.vstack(out)


def thin_by_spacing(xy, min_spacing):
    """Drop breadcrumbs closer than min_spacing to the last kept one.

    Not a plain stride: the block is motionless for much of a demo (the pusher
    repositions), so strided samples would stack dozens of coincident dots.
    """
    kept = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[kept[-1]]) >= min_spacing:
            kept.append(i)
    return xy[kept]


def transport_points(points_mv, M, Mr):
    """Move CGA points by the motor sandwich, returning euclidean (N, 2).

    The sandwich is collapsed to one 16x16 matrix first (it is linear in the
    multivector), which is ~20x faster than sandwiching every point directly.
    A unit motor keeps points normalised, so e1/e2 read straight off blades 1, 2.
    """
    sandwich = even_sandwhich(np.eye(16), M, Mr)
    return (points_mv @ sandwich)[:, 1:3]


def screw_interp(M0, M0r, M1, M1r, tau):
    """Motor a fraction tau along the screw motion carrying M0 to M1.

    Returns the motor pair, not a pose: the caller needs it to transport the
    block's CGA points, and extract_pose recovers the frame from it.
    """
    DM, DMr = gp(M1, M0r), gp(M0, M1r)
    t_rel, dtheta = extract_pose(DM, DMr)

    if abs(np.sin(dtheta / 2)) < 1e-9:
        # pure translation: the pivot runs off to infinity
        t = tau * np.asarray(t_rel)
        DM_tau, DM_tau_r = translation(*t), translation_rev(*t)
    else:
        c_, s_ = np.cos(dtheta), np.sin(dtheta)
        pivot = np.linalg.solve(np.eye(2) - np.array([[c_, -s_], [s_, c_]]), t_rel)
        Tc, Tcr = translation(*pivot), translation_rev(*pivot)
        Tnc, Tncr = translation(*(-pivot)), translation_rev(*(-pivot))
        Rt, Rtr = rotation(tau * dtheta), rotation_rev(tau * dtheta)
        DM_tau = gp(Tc, gp(Rt, Tnc))
        DM_tau_r = gp(Tncr, gp(Rtr, Tcr))

    return gp(DM_tau, M0), gp(M0r, DM_tau_r)


SHADERS = {
    "points_vertex": "visual/shaders/point.vert",
    "points_fragment": "visual/shaders/point.frag",

    "circles_vertex": "visual/shaders/circle.vert",
    "circles_fragment": "visual/shaders/circle.frag",

    "lines_vertex": "visual/shaders/line.vert",
    "lines_fragment": "visual/shaders/line.frag",
}


def shader_src(key):
    return open(SHADERS[key]).read()


def new_points():
    return Point(vertex_src=shader_src("points_vertex"),
                 fragment_src=shader_src("points_fragment"))


def new_frame(axis_length):
    return Frame(vertex_src=shader_src("lines_vertex"),
                 fragment_src=shader_src("lines_fragment"),
                 axis_length=axis_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demos", nargs="*", type=int,
                        help="demo indices to render (default: best multimodal group)")
    parser.add_argument("-k", "--group", type=int, default=5,
                        help="auto-picked group size; data thins out past ~6 (max ~8)")
    parser.add_argument("-f", "--dataset", default=DATASET, type=Path)
    parser.add_argument("--px-per-unit", type=float, default=None,
                        help="zoom override; default fits the scene to the window")
    parser.add_argument("--spacing", type=float, default=BREADCRUMB_MIN_GAP,
                        help="min breadcrumb gap in world units (0 = every sample)")
    parser.add_argument("--no-tee", action="store_true",
                        help="draw only pose frames, not the block outline")
    args = parser.parse_args()

    all_demos, dt = load_demos(args.dataset)

    if args.demos:
        demo_ids = args.demos
    else:
        demo_ids, spread, mean_sep = find_multimodal_group(all_demos, args.group)
        print(f"auto-picked multimodal group of {len(demo_ids)}: demos {demo_ids}\n"
              f"  starts within {spread:.1f} px, paths diverge by {mean_sep:.1f} px "
              f"on average (ratio {mean_sep / max(spread, 1.0):.2f})")

    poses_px = [all_demos[i] for i in demo_ids]
    px_per_unit = args.px_per_unit or fit_px_per_unit(poses_px, with_block=not args.no_tee)
    poses = [to_world(p, px_per_unit) for p in poses_px]
    goal_pose = to_world(np.array([GOAL_POSE_PX]), px_per_unit)[0]

    print(f"scale: {px_per_unit:.1f} px/unit")
    for i, pose in zip(demo_ids, poses):
        print(f"  demo {i:3d}  T={len(pose):3d}  "
              f"start={np.round(pose[0], 2)}  end={np.round(pose[-1], 2)}")
    print("view: scroll to zoom about the cursor (+/- keys too), drag to pan, 0 to reset")

    # one motor per recorded sample; playback only interpolates between neighbours
    pose_motors = [[motor_of(th, (x, y)) for x, y, th in pose] for pose in poses]

    # The block outline as CGA points in the body frame: the same motor that
    # defines the pose transports them by a sandwich product, so the drawn shape
    # and the drawn frame are literally the same rigid motion.
    outline_mv = point_to_mv(
        sample_polygon(body_to_world(TEE_OUTLINE_PX, px_per_unit), OUTLINE_SAMPLE_STEP))

    window = Window(shader_paths_dict=SHADERS)
    window.dyn_renderers.clear()   # drop the stock untinted renderers

    # thin per-path decoration as the group grows, else it becomes a mat of dots
    n_demos = len(poses)
    n_breadcrumb_frames = max(3, round((14 if not args.no_tee else 28) / n_demos))
    path_dot_size = 8 if n_demos <= 3 else (6 if n_demos <= 6 else 5)

    animated = []
    for index, pose in enumerate(poses):
        style = demo_style(index, n_demos)

        path_dots = new_points()
        path_dots.update([tuple(q) for q in thin_by_spacing(pose[:, :2], args.spacing)])
        window.dyn_renderers[f"path{index}"] = StyledRenderer(
            path_dots, color=style["dots"], size=path_dot_size)

        idx = np.unique(np.linspace(0, len(pose) - 1, n_breadcrumb_frames).round().astype(int))
        breadcrumb_frames = new_frame(BREADCRUMB_FRAME_AXIS_LEN)
        breadcrumb_frames.update([tuple(q) for q in pose[idx]])
        window.dyn_renderers[f"breadcrumbs{index}"] = StyledRenderer(
            breadcrumb_frames, x_color=style["x"], y_color=style["y"], width=1.5)

        live_frame = new_frame(MOVING_FRAME_AXIS_LEN)
        window.dyn_renderers[f"live{index}"] = StyledRenderer(
            live_frame, x_color=style["x"], y_color=style["y"], width=3.0)

        live_outline = None
        if not args.no_tee:
            live_outline = new_points()
            window.dyn_renderers[f"tee{index}"] = StyledRenderer(
                live_outline, color=style["dots"], size=OUTLINE_DOT_SIZE)
        animated.append((live_frame, live_outline))

    goal_frame = new_frame(MOVING_FRAME_AXIS_LEN)
    goal_frame.update([tuple(goal_pose)])
    window.dyn_renderers["goal"] = StyledRenderer(
        goal_frame, x_color=GOAL_STYLE["x"], y_color=GOAL_STYLE["y"], width=2.0)

    # the target drawn as the block itself, so what the demos converge on reads as a T
    if not args.no_tee:
        goal_outline = new_points()
        goal_outline.update(transport_points(
            outline_mv, *motor_of(goal_pose[2], tuple(goal_pose[:2]))))
        window.dyn_renderers["teegoal"] = StyledRenderer(
            goal_outline, color=GOAL_STYLE["x"], size=OUTLINE_DOT_SIZE)

    # Phase-locked playback: every demo runs 0 -> 1 over the same wall-clock span.
    # They have different lengths, and letting one finish early hides the divergence.
    duration = max(len(p) for p in poses) * dt / PLAYBACK_SPEED
    tau, tau_dir = 0.0, 1
    last = glfw.get_time()

    while window.is_running():
        now = glfw.get_time()
        tau += tau_dir * (now - last) / duration
        last = now
        if tau >= 1.0:
            tau, tau_dir = 1.0, -1
        elif tau <= 0.0:
            tau, tau_dir = 0.0, 1

        for pose, motors, (live_frame, live_outline) in zip(poses, pose_motors, animated):
            phase = tau * (len(pose) - 1)
            i = min(int(phase), len(pose) - 2)
            M, Mr = screw_interp(*motors[i], *motors[i + 1], phase - i)
            position, theta = extract_pose(M, Mr)
            live_frame.update([(position[0], position[1], theta)])
            if live_outline is not None:
                live_outline.update(transport_points(outline_mv, M, Mr))

        window.step()


if __name__ == "__main__":
    main()
