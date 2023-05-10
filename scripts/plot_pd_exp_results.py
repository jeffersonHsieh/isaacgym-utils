import pandas as pd


df = pd.read_csv("pick_block_log.txt",names=['file','Kp','Kd','Error'])

# plot lines on one chart, one K_p value per line, 
# use K_d as x-axis, Error as y-axis
# use K_p as legend

df = df.pivot(index="Kd",columns="Kp",values="Error")
ax = df.plot(
    title="PD gain vs End Effector Final Position Error (2-norm)",
    logy=True,xlabel="Kd",
    ylabel="End Effector Pos Error (m)"
    )

fig = ax.get_figure()
fig.savefig("pick-block-hparam.png")

