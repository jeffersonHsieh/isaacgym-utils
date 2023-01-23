import numpy as np
import sys

# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

GET_PATH = "/home/mark/github/isaacgym-utils/scripts/IP_dataset/"
# GET_PATH = "/home/mark/github/isaacgym-utils/scripts/dataset/"
tree_start = 0
tree_num = 1

tree_pts = 9

NUM_ENV =  100
NUM_K_VAL = 47


K_min = 5
K_max = 235

K = np.linspace(K_min, K_max, NUM_K_VAL).astype(int)

F = np.linspace(1, 100, NUM_ENV).astype(int)

def main():
    print(f" ================ starting sample script ================  ")
    y_vert_arrays = []
    x_vert_arrays = []
    force_applied_arrays = []


    print(f"NUM_ENV: {NUM_ENV}, NUM_K_values: {NUM_K_VAL}  ")
    FxK_matrix = np.zeros((NUM_ENV, NUM_K_VAL))

    for k_idx, k_val in enumerate (K):

        prefix = "[%s]"%tree_pts

        for env in range(0, NUM_ENV):
            init_pos = np.load(GET_PATH + prefix + 'X_vertex_init_pos_treeK%s_env%s.npy'%(k_val, env))
            final_pos = np.load(GET_PATH + prefix + 'Y_vertex_final_pos_treeK%s_env%s.npy'%(k_val, env))
            # print(GET_PATH + prefix + 'Y_vertex_final_pos_treeK%s_env%s.npy'%(k_val, env))
            #get X displacement 
            init_pos = init_pos[0,0:3,:] #final pos shape 1,7,9 -> 3,9
            final_pos = final_pos[0,0:3,:] #final pos shape 1,7,9 -> 3,9

            delta_pos = final_pos - init_pos
            delta_pos_norm = np.linalg.norm(delta_pos)

            FxK_matrix[env,k_idx] = delta_pos_norm
  


    print(f" shape of FxK_matrix: {FxK_matrix.shape}")
    FxK_heatmap = pd.DataFrame(FxK_matrix,columns=K,index=F)

    #Plot confusion matrix heatmap
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    

    ax = sns.heatmap(FxK_heatmap,
            cmap='coolwarm',   cbar_kws={'label': 'displacement (m)'})
    ax.invert_yaxis()

    plt.xlabel('K',fontsize=22)
    plt.ylabel('F',fontsize=22)

    plt.show()
    






    print(f" ================ ending sample script ================  ")


if __name__ == '__main__':
    main()

