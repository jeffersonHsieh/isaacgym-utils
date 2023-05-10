import numpy as np
import sys
from pathlib import Path

if __name__=="__main__":
    assert len(sys.argv)==2

    path = Path(sys.argv[1])
    fname = path.stem
    outdir = path.parent / fname
    outdir.mkdir(exist_ok=True)

    trajectories = np.load(path,allow_pickle=True)
    trajectories = list(map(np.array,trajectories))

    for i,trajectory in enumerate(trajectories):
        np.save(outdir / f"{fname}_traj_{i}.npy",trajectory)
