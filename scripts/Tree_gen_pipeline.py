import math
import random

import SCA_tree_gen as sca

TREE_NUM = 20
MAX_TREE_POINTS = 100
TRUNK_HEIGHT_FACTORS = [1,2,3]
SCALING = 5
PIPE_MODEL_EXPONENT = 2 #suggested values: 2 or 3
MAX_ATTRACTION_POINTS = 200 #lower values lead to more irregular branches
STEP_WIDTH = 1 #determines the lenght of individual tree segments
a = 10 # suggested values Trunk_Height and higher
HEIGHT_STRECH_VALS = [1, 2, 0.5, 1.5, 0.33]
WIDTH_STRECH_VALS = [1, 2, 0.5, 1.5, 0.33]
ATT_PTS_NUM = [100, 200, 400, 800, 1600]

tree = 0
while tree < TREE_NUM:
    trunk_height = STEP_WIDTH * TRUNK_HEIGHT_FACTORS[random.randrange(0, len(TRUNK_HEIGHT_FACTORS))] / SCALING
    d_termination = STEP_WIDTH/random.randrange(1, 6)
    d_attraction_values = [math.ceil(trunk_height)+1, math.ceil(trunk_height) + 2, math.ceil(trunk_height) + 4, math.ceil(trunk_height) + 8, math.ceil(trunk_height) + 16, math.ceil(trunk_height) + 32, math.ceil(trunk_height) + 64]
    d_attraction = d_attraction_values[random.randrange(0, len(d_attraction_values) - 1)]
    height_strech = HEIGHT_STRECH_VALS[random.randrange(0,len(HEIGHT_STRECH_VALS)-1)]
    width_strech = WIDTH_STRECH_VALS[random.randrange(0,len(WIDTH_STRECH_VALS)-1)]
    att_pts_max = ATT_PTS_NUM[random.randrange(0, len(ATT_PTS_NUM)-1)]
    print("tree%s: \n\t d_termination: %s \n\t d_attraction: %s \n\t height_strech: %s \n\t width_strech: %s \n\t att_pts_max: %s"%(tree, d_termination, d_attraction, height_strech, width_strech, att_pts_max))
    tg = sca.TreeGenerator(max_steps=10000, att_pts_max=att_pts_max, da=d_attraction, dt=d_termination, step_width=STEP_WIDTH, offset=[-0.5, -0.5, trunk_height], scaling=SCALING, max_tree_points=MAX_TREE_POINTS, tip_radius=0.1, tree_id=tree, pipe_model_exponent=PIPE_MODEL_EXPONENT, z_strech=height_strech, y_strech=width_strech, x_strech=width_strech, step_width_scaling=0.8)
    tg.generate_tree()
    tg.calculate_branch_thickness()
    tg.generate_urdf()
    tree+=1