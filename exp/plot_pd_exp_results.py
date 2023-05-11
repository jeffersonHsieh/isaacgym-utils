import pandas as pd
from pathlib import Path

import sys
path = Path(sys.argv[1])
df = pd.read_csv(str(path),names=['file','Kp','Kd','Error'])

# plot lines on one chart, one K_p value per line, 
# use K_d as x-axis, Error as y-axis
# use K_p as legend

df = df.pivot(index="Kd",columns="Kp",values="Error")
ax = df.plot(
    title="PD gain vs End Effector Final Position Error (2-norm)",
    # logx=True,
    logy=True,xlabel="Kd",
    ylabel="End Effector Pos Error (m)"
    )

fig = ax.get_figure()
fig.savefig(f"{path.stem}.png")

