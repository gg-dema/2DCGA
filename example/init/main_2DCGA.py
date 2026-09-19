import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from src.mv_embedding import *
from src.CGA_2D_op import *
from visual.Window import Window


shaders_paths = {
        "points_vertex": "visual/shaders/point.vert",
        "points_fragment": "visual/shaders/point.frag",

        "circles_vertex": "visual/shaders/circle.vert",
        "circles_fragment": "visual/shaders/circle.frag",

        "lines_vertex": "visual/shaders/line.vert",
        "lines_fragment": "visual/shaders/line.frag",  
    }

window = Window(shader_paths_dict=shaders_paths)

points = np.array([
    [0.0, 1.5],
    [-1., -0.5],
    [1., -0.5],
])
MV_points = point_to_mv(points)


n = np.array([
    [0.0, 1.0]
])
d = np.array([
    [0.5]
    ])
MV_line = dual_line_mv(n, d)



C_dual = dual_circle_from_CR(
    np.array([
        [1.0, 0.0]
    ]),
    radius=2.50
)

theta = 0.01
R        = rotation(theta)
R_rev    = rotation_rev(theta)

ln_lambda = -0.01
D          = dilatation(ln_lambda)  
D_rev      = dilatation_rev(ln_lambda)

tx, ty = 0.01, 0.01
T          = translation(tx, ty)
T_rev      = translation_rev(tx, ty)


i = 0

while window.is_running():

    MV_points = even_sandwhich(MV_points, R, R_rev)
    MV_line = even_sandwhich(MV_line, R, R_rev)
    C_dual = even_sandwhich(C_dual, R, R_rev)

    if i % 50 == 0:
        tx *= -1
        ty *= -1
        T  = translation(tx, ty)
        T_rev   = translation_rev(tx, ty)

        ln_lambda *= -1
        D  = dilatation(ln_lambda)
        D_rev   = dilatation_rev(ln_lambda)

    if i % 100 == 0:
        theta *= -1
        R  = rotation(theta)
        R_rev   = rotation_rev(theta)

    MV_points = even_sandwhich(MV_points, D, D_rev)
    MV_line = even_sandwhich(MV_line, D, D_rev)
    C_dual = even_sandwhich(C_dual, D, D_rev)

    MV_points = even_sandwhich(MV_points, T, T_rev)
    MV_line = even_sandwhich(MV_line, T, T_rev)
    C_dual = even_sandwhich(C_dual, T, T_rev)

    euclidean_points = mv_to_point(MV_points)
    x0, v = euc_line_from_dual_line_mv(MV_line) # each (N, 2)
    xc, r = center_radius_from_dual_circle(C_dual)
    window.dyn_renderers['lines'].update(list(zip(x0, v)))
    window.dyn_renderers['points'].update(euclidean_points)
    window.dyn_renderers['circles'].update(np.hstack((xc.squeeze(), r.squeeze())))
    window.step()
    i+=1