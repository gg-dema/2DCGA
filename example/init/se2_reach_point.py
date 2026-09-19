import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import numpy as np
from src.CGA_2D_op import *
from src.mv_embedding import *
from visual.Window import Window


# start pose
theta0, t0 = 0.3, (5.0, 0.5)
M0, M0r = motor_of(theta0, t0)

# goal: reach point (px,py) with some chosen arrival heading theta1
px, py = -2.0, 3.0
theta1 = 1.8
M1, M1r = motor_of(theta1, (px, py))

# relative motor Delta M = M1 * M0^-1
DM  = gp(M1, M0r)
DMr = gp(M0, M1r)

t_rel, dtheta = extract_pose(DM, DMr)
print('relative translation t:', t_rel, ' relative angle dtheta:', dtheta)

# center of rotation c = (I - R(dtheta))^-1 @ t_rel
c_, s_ = np.cos(dtheta), np.sin(dtheta)
Imat = np.eye(2)
Rmat = np.array([[c_, -s_],[s_, c_]])
c = np.linalg.solve(Imat - Rmat, t_rel)
print('center of rotation c:', c)

neg_c = -c
Tc, Tcr = translation(*c), translation_rev(*c)
Tnc, Tncr = translation(*neg_c), translation_rev(*neg_c)

def motor_at(tau):
    # pure rotation by tau*dtheta about the fixed pivot c, applied on top of M0
    Rt, Rtr = rotation(tau * dtheta), rotation_rev(tau * dtheta)

    DM_tau   = gp(Tc, gp(Rt, Tnc))
    DM_tau_r = gp(Tncr, gp(Rtr, Tcr))

    M_tau   = gp(DM_tau, M0)
    M_tau_r = gp(M0r, DM_tau_r)
    return extract_pose(M_tau, M_tau_r)


shaders_paths = {
    "points_vertex": "visual/shaders/point.vert",
    "points_fragment": "visual/shaders/point.frag",

    "circles_vertex": "visual/shaders/circle.vert",
    "circles_fragment": "visual/shaders/circle.frag",

    "lines_vertex": "visual/shaders/line.vert",
    "lines_fragment": "visual/shaders/line.frag",
}
window = Window(shader_paths_dict=shaders_paths)

# static context: target point and the fixed pivot of the screw motion
window.dyn_renderers['points'].update([(px, py), tuple(c)])

tau, tau_dir = 0.0, 1
tau_step = 0.005

while window.is_running():
    pos, theta = motor_at(tau)

    # current frame (moving) + goal frame (static), same renderer
    window.dyn_renderers['frames'].update([
        (pos[0], pos[1], theta),
        (px, py, theta1),
    ])

    # ping-pong tau between 0 and 1 so the motion loops back and forth
    tau += tau_dir * tau_step
    if tau >= 1.0:
        tau, tau_dir = 1.0, -1
    elif tau <= 0.0:
        tau, tau_dir = 0.0, 1

    window.step()