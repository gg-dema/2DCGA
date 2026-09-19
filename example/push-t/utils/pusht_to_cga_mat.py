"""
Re-encode datasets/PushT_SE2/PushT.mat as raw CGA multivectors.

Every trajectory sample becomes a motor M (16 floats) with
    P_world = M P_body M~
so the pose is carried as one object instead of an (x, y, theta) triple, and the
same sandwich transports points, circles and lines alike.

Frame is unchanged from the source file: raw Push-T pixels, y pointing down,
NOT the mirrored/rescaled world the renderer uses. This file is the same data in
a different algebra, not a different dataset.

Blade order (src/CGA_2D.py), index = bitmask over (e1, e2, ep, en):
    0: 1      4: ep        8: en         12: ep^en
    1: e1     5: e1^ep     9: e1^en      13: e1^ep^en
    2: e2     6: e2^ep    10: e2^en      14: e2^ep^en
    3: e1^e2  7: e1^e2^ep 11: e1^e2^en   15: e1^e2^ep^en
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from src.CGA_2D_op import *
from src.mv_embedding import *

SRC = Path("/home/dema/Projects/RobLearning/Squid_v2/datasets/PushT_SE2/PushT.mat")
DST = Path(__file__).resolve().parent / "datasets" / "PushT.cga.mat"

GOAL_POSE_PX = (256.0, 256.0, np.pi / 4)
BODY_KP_PX = np.array([
    [  9.27,  90.04], [-60.98,  -1.19], [ 60.91,  -1.39],
    [ -0.27,  25.54], [-14.07, 122.82], [ 39.00,  32.86],
    [-39.97,  32.73], [-16.82,  62.86], [ 23.54,  -1.70],
])
TEE_SCALE, TEE_LEN = 30.0, 4
TEE_OUTLINE_PX = np.array([
    (-TEE_LEN * TEE_SCALE / 2, 0.0), (TEE_LEN * TEE_SCALE / 2, 0.0),
    (TEE_LEN * TEE_SCALE / 2, TEE_SCALE), (TEE_SCALE / 2, TEE_SCALE),
    (TEE_SCALE / 2, TEE_LEN * TEE_SCALE), (-TEE_SCALE / 2, TEE_LEN * TEE_SCALE),
    (-TEE_SCALE / 2, TEE_SCALE), (-TEE_LEN * TEE_SCALE / 2, TEE_SCALE),
])
BLADES = ["1", "e1", "e2", "e1^e2", "ep", "e1^ep", "e2^ep", "e1^e2^ep",
          "en", "e1^en", "e2^en", "e1^e2^en", "ep^en", "e1^ep^en",
          "e2^ep^en", "e1^e2^ep^en"]


def motors_of(pose):
    """(T, 3) [x, y, theta] -> (16, T) motor and (16, T) reverse."""
    M = np.empty((len(pose), 16))
    Mr = np.empty((len(pose), 16))
    for i, (x, y, th) in enumerate(pose):
        M[i], Mr[i] = motor_of(th, (x, y))
    return M.T, Mr.T


if __name__ == "__main__":
    src = loadmat(str(SRC))['demos']
    n = src.shape[1]

    dtype = np.dtype([('motor', object), ('motor_rev', object),
                      ('pos', object), ('dt', object)])
    demos = np.empty((1, n), dtype=object)
    for j in range(n):
        pose = src[0, j]['pos'][0, 0].T.astype(float)      # (T, 3)
        dt = src[0, j]['dt'][0, 0]
        M, Mr = motors_of(pose)
        inner = np.empty((1, 1), dtype=dtype)
        inner[0, 0]['motor'] = M                            # (16, T)
        inner[0, 0]['motor_rev'] = Mr
        inner[0, 0]['pos'] = pose.T                         # (3, T), as in the source
        inner[0, 0]['dt'] = dt
        demos[0, j] = inner

    gM, gMr = motor_of(GOAL_POSE_PX[2], GOAL_POSE_PX[:2])
    out = {
        "demos": demos,
        "goal_motor": gM[:, None],
        "goal_motor_rev": gMr[:, None],
        "goal_pose": np.array(GOAL_POSE_PX)[:, None],
        "keypoints": point_to_mv(BODY_KP_PX).T,             # (16, 9) CGA null points
        "keypoints_euc": BODY_KP_PX.T,                      # (2, 9)
        "outline": point_to_mv(TEE_OUTLINE_PX).T,           # (16, 8)
        "outline_euc": TEE_OUTLINE_PX.T,                    # (2, 8)
        "blades": np.array(BLADES, dtype=object),
        "metric": np.array([1.0, 1.0, 1.0, -1.0])[:, None],
        "frame": "raw Push-T pixels, y down; same convention as PushT.mat",
    }
    savemat(str(DST), out)
    print(f"wrote {DST}  ({DST.stat().st_size/1e6:.1f} MB)  demos={n}")
