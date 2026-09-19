"""
Export the Push-T end-effector track and contact flag as a LASA-style .mat.

    demos[0, j]['agent']      (2, T)  pusher centre, raw Push-T pixels
    demos[0, j]['n_contacts'] (1, T)  0, 1 or 2 -- contacts reported by pymunk
    demos[0, j]['dt']         scalar

Rows align 1:1 with datasets/PushT_SE2/PushT.mat, so demo j sample t here is the
same instant as demo j sample t there.

n_contacts is a COUNT, not a position: the replay buffer stores no contact
coordinates. It reads 2 when the pusher straddles the junction of the T's two
constituent boxes. It is trustworthy as a touch flag -- when it is >= 1 the
distance from the pusher centre to the T surface has median 14.88 px against the
pusher's 15 px radius.

Source is the Diffusion Policy replay buffer, which is NOT checked in:
    https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
Pass the unzipped pusht_cchi_v7_replay.zarr as argv[1]. Needs `zarr`, which the
2DCGA env does not have -- generate once, then only the .mat is needed.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

AGENT_RADIUS_PX = 15.0      # PushTEnv.add_circle((256, 400), 15)
DT = 0.1

if __name__ == "__main__":
    import zarr

    zarr_path = Path(sys.argv[1])
    dst = Path(__file__).resolve().parent / "datasets" / "PushT.ee.mat"

    root = zarr.open(str(zarr_path), "r")
    state = np.asarray(root["data/state"][:], dtype=np.float64)
    ncont = np.asarray(root["data/n_contacts"][:], dtype=np.float64).reshape(-1)
    ends = root["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])

    dtype = np.dtype([('agent', object), ('n_contacts', object), ('dt', object)])
    demos = np.empty((1, len(ends)), dtype=object)
    for j, (s, e) in enumerate(zip(starts, ends)):
        inner = np.empty((1, 1), dtype=dtype)
        inner[0, 0]['agent'] = state[s:e, 0:2].T            # (2, T)
        inner[0, 0]['n_contacts'] = ncont[s:e][None, :]     # (1, T)
        inner[0, 0]['dt'] = np.array([[DT]])
        demos[0, j] = inner

    savemat(str(dst), {
        "demos": demos,
        "agent_radius": np.array([[AGENT_RADIUS_PX]]),
        "frame": "raw Push-T pixels, y down; rows align 1:1 with PushT_SE2/PushT.mat",
    })
    print(f"wrote {dst}  ({dst.stat().st_size/1e6:.1f} MB)  demos={len(ends)}  "
          f"contact in {100*(ncont>0).mean():.1f}% of steps")
