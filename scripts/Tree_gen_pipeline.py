import math
import random
import multiprocessing
import SCA_tree_gen as sca
import franka_import_tree_multi_env as fit

TREE_NUM = 100
ENV_NUM = 100
MAX_TREE_POINTS = 10
TRUNK_HEIGHT_FACTORS = [1,2]
SCALING = 2
PIPE_MODEL_EXPONENT = 3 #suggested values: 2 or 3
MAX_ATTRACTION_POINTS = 200 #lower values lead to more irregular branches
STEP_WIDTH = 1 #determines the lenght of individual tree segments
a = 10 # suggested values Trunk_Height and higher
HEIGHT_STRECH_VALS = [0.5, 0.33]
WIDTH_STRECH_VALS = [1, 1.5, 2, 3]
ATT_PTS_NUM = [200, 400, 800, 1600]

yaml_paths = []
urdf_paths = []
name_dicts = []
edge_defs = []

parser = argparse.ArgumentParser()
parser.add_argument("-tree_pts", type=int, dest="tree_pts", help="number of generated tree points")
args = parser.parse_args()
tree_pts = args.tree_pts

tree = 0
while tree < TREE_NUM:
    trunk_height = STEP_WIDTH * 0.75 / SCALING #TRUNK_HEIGHT_FACTORS[random.randrange(0, len(TRUNK_HEIGHT_FACTORS))] / SCALING
    d_termination = STEP_WIDTH/random.randrange(3, 6)
    d_attraction_values = [math.ceil(trunk_height)+1, math.ceil(trunk_height) + 2, math.ceil(trunk_height) + 4, math.ceil(trunk_height) + 8, math.ceil(trunk_height) + 16, math.ceil(trunk_height) + 32, math.ceil(trunk_height) + 64]
    d_attraction = d_attraction_values[random.randrange(0, len(d_attraction_values) - 1)]
    height_strech = HEIGHT_STRECH_VALS[random.randrange(0,len(HEIGHT_STRECH_VALS)-1)]
    width_strech = WIDTH_STRECH_VALS[random.randrange(0,len(WIDTH_STRECH_VALS)-1)]
    att_pts_max = ATT_PTS_NUM[random.randrange(0, len(ATT_PTS_NUM)-1)]
    print("tree%s: \n\t d_termination: %s \n\t d_attraction: %s \n\t height_strech: %s \n\t width_strech: %s \n\t att_pts_max: %s"%(tree, d_termination, d_attraction, height_strech, width_strech, att_pts_max))
    tg = sca.TreeGenerator(max_steps=10000, att_pts_max=att_pts_max, da=d_attraction, dt=d_termination, step_width=STEP_WIDTH, offset=[-0.5, -0.5, trunk_height], scaling=SCALING, max_tree_points=tree_pts, tip_radius=0.1, tree_id=tree, pipe_model_exponent=PIPE_MODEL_EXPONENT, z_strech=height_strech, y_strech=width_strech, x_strech=width_strech, step_width_scaling=0.65, env_num = ENV_NUM)
    tg.generate_tree()
    tg.calculate_branch_thickness()
    name_dict, edge_def, urdf_path = tg.generate_urdf()
    #urdf_paths.append(urdf_path)
    #name_dicts.append(name_dict)
    #urdf_path = "/home/jan-malte/Tree_Deformation_Project/isaacgym-utils/scripts/test1.urdf"
    #name_dict = {'joints': ['joint0_x', 'joint0_y', 'joint0_z_to_1', 'joint1_x', 'joint1_y', 'joint1_z_to_2', 'joint2_x', 'joint2_y'], 'links': ['base_link', 'link_0_to_1', 'link_1_to_2']}
    yaml_path, stiffness_list, damping_list = tg.generate_yaml()
    #yaml_paths.append(yaml_path)
    #yaml_path = "/home/jan-malte/Tree_Deformation_Project/isaacgym-utils/scripts/test1.yaml"
    edge_def2 = tg.calc_edge_tuples()
    #edge_defs.append(edge_def)
    #edge_def = [(0, 1), (1, 2)]
    #print("name joints: " + str(len(name_dict["joints"])))
    #print("name links: " + str(len(name_dict["links"])))
    print(edge_def)
    print(edge_def2)

    fit.import_tree(name_dict, urdf_path, yaml_path, edge_def, stiffness_list, damping_list, tree_num=tree, tree_pts=tree_pts)

    tree+=1

#fit.import_tree(name_dicts, urdf_paths, yaml_paths, edge_defs, tree_num=0)