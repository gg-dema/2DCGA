"""
Push-T playback with the pusher (end effector) and its contact point.

se2_pusht_demo.py shows what the T block did; this adds what moved it. The
pusher is the one genuine circle in the scene -- pymunk gives it a real radius --
so it is carried as a CGA dual circle and moved by a translator sandwich.

The replay buffer stores no contact coordinates, only data/n_contacts (0, 1 or 2;
the T is two boxes, so straddling their junction reads 2). The count gates the
contact dot, and its position is derived as the closest point on the T outline.

Everything else is imported from se2_pusht_demo.
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import glfw
import numpy as np
from scipy.io import loadmat

import se2_pusht_demo as base
from se2_pusht_demo import (
    BREADCRUMB_FRAME_AXIS_LEN, BREADCRUMB_MIN_GAP, GOAL_POSE_PX, GOAL_STYLE,
    MOVING_FRAME_AXIS_LEN, OUTLINE_DOT_SIZE, OUTLINE_SAMPLE_STEP, PLAYBACK_SPEED,
    SHADERS, TEE_OUTLINE_PX, StyledRenderer, body_to_world, demo_style, extract_pose,
    find_multimodal_group, load_demos, motor_of, new_frame, new_points, sample_polygon,
    screw_interp, shader_src, thin_by_spacing, to_world, transport_points)
from src.CGA_2D_op import even_sandwhich, translation, translation_rev
from src.mv_embedding import (point_to_mv, dual_circle_from_CR,
                              center_radius_from_dual_circle)
from visual.Window import Window, PIXELS_PER_UNIT
from visual.objects.circles import Circle

EE_DATASET = Path(__file__).resolve().parents[2] / "datasets" / "PushT.ee.mat"

# circle.frag cuts the ring's inside at dist < 1 - thickness; a huge thickness
# makes that test never fire, so the ring fills solid. No shader change needed.
FILLED_RING = 1.0e6
PUSHER_ALPHA = 0.35        # the pusher overlaps the block; keep the outline readable
CONTACT_DOT_SIZE = 9       # px
PUSHER_CLIP_PX = 90.0      # see fit_px_per_unit_with_pusher


def load_pusher_tracks(path):
    """demos[0, j] -> {'agent': (2, T), 'n_contacts': (1, T)}, plus the disc radius."""
    f = loadmat(str(path))
    d = f['demos']
    agents = [d[0, j]['agent'][0, 0].T.astype(float) for j in range(d.shape[1])]
    contacts = [d[0, j]['n_contacts'][0, 0].ravel().astype(float) for j in range(d.shape[1])]
    return agents, contacts, float(f['agent_radius'][0, 0])


def fit_px_per_unit_with_pusher(poses_px, pusher_px):
    """Auto-fit that keeps the pusher on screen without shrinking the block.

    The pusher wanders far while repositioning (p99 gap 280 px, block only 120 px
    across), so folding its raw track into the bounds costs a 1.23x shrink for a
    few frames. Clipping each sample to PUSHER_CLIP_PX of its block pose costs 1.05x.
    """
    clipped = []
    for pose, pusher in zip(poses_px, pusher_px):
        offset = pusher - pose[:, :2]
        dist = np.linalg.norm(offset, axis=1, keepdims=True)
        clipped.append(pose[:, :2]
                       + offset * np.minimum(1.0, PUSHER_CLIP_PX / np.maximum(dist, 1e-9)))
    xy = np.vstack([np.vstack(clipped)] + [p[:, :2] for p in poses_px])
    return max(base.fit_px_per_unit(poses_px),
               (xy.max(0) - xy.min(0)).max() / base.FIT_WORLD_UNITS)


def closest_on_polygon(point, polygon):
    """Closest point to `point` on a closed polygon's boundary."""
    a = polygon
    ab = np.roll(polygon, -1, axis=0) - a
    t = np.clip(((point - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    foot = a + t[:, None] * ab
    return foot[np.argmin(np.linalg.norm(point - foot, axis=1))]


def new_circle():
    return Circle(vertex_src=shader_src("circles_vertex"),
                  fragment_src=shader_src("circles_fragment"),
                  pixels_per_unit=PIXELS_PER_UNIT)


def translated_circle(circle_mv, xy):
    """The dual circle moved to xy by a translator sandwich -> (x, y, radius)."""
    moved = even_sandwhich(circle_mv, translation(*xy), translation_rev(*xy))
    center, radius = center_radius_from_dual_circle(moved)
    return np.column_stack([center, radius])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demos", nargs="*", type=int,
                        help="demo indices to render (default: best multimodal group)")
    parser.add_argument("-k", "--group", type=int, default=3,
                        help="auto-picked group size; denser than se2_pusht_demo, so lower")
    parser.add_argument("-f", "--dataset", default=base.DATASET, type=Path)
    parser.add_argument("--ee-dataset", default=EE_DATASET, type=Path)
    parser.add_argument("--px-per-unit", type=float, default=None)
    parser.add_argument("--spacing", type=float, default=BREADCRUMB_MIN_GAP)
    parser.add_argument("--no-tee", action="store_true")
    args = parser.parse_args()

    all_demos, dt = load_demos(args.dataset)
    all_pushers, all_contacts, pusher_radius_px = load_pusher_tracks(args.ee_dataset)
    if len(all_pushers) != len(all_demos):
        raise SystemExit("EE dataset and pose dataset disagree on demo count")

    if args.demos:
        demo_ids = args.demos
    else:
        demo_ids, spread, mean_sep = find_multimodal_group(all_demos, args.group)
        print(f"auto-picked multimodal group of {len(demo_ids)}: demos {demo_ids}\n"
              f"  starts within {spread:.1f} px, paths diverge by {mean_sep:.1f} px on average")

    poses_px = [all_demos[i] for i in demo_ids]
    pusher_px = [all_pushers[i] for i in demo_ids]
    contacts = [all_contacts[i] for i in demo_ids]
    for pose, pusher in zip(poses_px, pusher_px):
        if len(pose) != len(pusher):
            raise SystemExit("EE track and pose track differ in length")

    px_per_unit = args.px_per_unit or fit_px_per_unit_with_pusher(poses_px, pusher_px)
    poses = [to_world(p, px_per_unit) for p in poses_px]
    pushers = [to_world(np.column_stack([p, np.zeros(len(p))]), px_per_unit)[:, :2]
               for p in pusher_px]
    goal_pose = to_world(np.array([GOAL_POSE_PX]), px_per_unit)[0]

    print(f"scale: {px_per_unit:.1f} px/unit   pusher radius {pusher_radius_px:.0f} px")
    for i, pose, contact in zip(demo_ids, poses, contacts):
        print(f"  demo {i:3d}  T={len(pose):3d}  "
              f"in contact {100 * (contact > 0).mean():.0f}% of steps")

    pose_motors = [[motor_of(th, (x, y)) for x, y, th in pose] for pose in poses]

    # Block geometry as CGA points: the outline for drawing, the bare polygon for
    # the contact query. Both ride the same motor as the pose frame.
    body_polygon = body_to_world(TEE_OUTLINE_PX, px_per_unit)
    outline_mv = point_to_mv(sample_polygon(body_polygon, OUTLINE_SAMPLE_STEP))
    polygon_mv = point_to_mv(body_polygon)

    # The pusher is a real disc in the sim, so it is a dual circle built at the
    # origin once and translated into place each frame.
    pusher_circle_mv = dual_circle_from_CR(np.zeros((1, 2)), pusher_radius_px / px_per_unit)

    window = Window(shader_paths_dict=SHADERS)
    window.dyn_renderers.clear()

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

        pusher_disc = new_circle()
        window.dyn_renderers[f"pusher{index}"] = StyledRenderer(
            pusher_disc, color=style["dots"][:3] + (PUSHER_ALPHA,), thickness_px=FILLED_RING)

        contact_dot = new_points()
        window.dyn_renderers[f"contact{index}"] = StyledRenderer(
            contact_dot, color=style["dots"], size=CONTACT_DOT_SIZE)

        animated.append((live_frame, live_outline, pusher_disc, contact_dot))

    goal_frame = new_frame(MOVING_FRAME_AXIS_LEN)
    goal_frame.update([tuple(goal_pose)])
    window.dyn_renderers["goal"] = StyledRenderer(
        goal_frame, x_color=GOAL_STYLE["x"], y_color=GOAL_STYLE["y"], width=2.0)

    if not args.no_tee:
        goal_outline = new_points()
        goal_outline.update(transport_points(
            outline_mv, *motor_of(goal_pose[2], tuple(goal_pose[:2]))))
        window.dyn_renderers["teegoal"] = StyledRenderer(
            goal_outline, color=GOAL_STYLE["x"], size=OUTLINE_DOT_SIZE)

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

        for pose, motors, pusher, contact, renderers in zip(
                poses, pose_motors, pushers, contacts, animated):
            live_frame, live_outline, pusher_disc, contact_dot = renderers

            phase = tau * (len(pose) - 1)
            i = min(int(phase), len(pose) - 2)
            M, Mr = screw_interp(*motors[i], *motors[i + 1], phase - i)
            position, theta = extract_pose(M, Mr)
            live_frame.update([(position[0], position[1], theta)])
            if live_outline is not None:
                live_outline.update(transport_points(outline_mv, M, Mr))

            # the pusher is a point mass, so plain interpolation is the honest choice
            pusher_now = pusher[i] + (phase - i) * (pusher[i + 1] - pusher[i])
            pusher_disc.update(translated_circle(pusher_circle_mv, pusher_now))

            # contact gated on the sim's own count, positioned by geometry
            if contact[int(round(phase))] > 0:
                touch = closest_on_polygon(pusher_now, transport_points(polygon_mv, M, Mr))
                contact_dot.update([tuple(touch)])
            else:
                contact_dot.update(np.empty((0, 2)))

        window.step()


if __name__ == "__main__":
    main()
