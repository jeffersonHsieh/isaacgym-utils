from franka_robot import *
import numpy as np
import quaternion
import matplotlib.pyplot as plt

def draw_box(ax, fbox_vertices):
    lines_idx_list = [[0, 1], [1, 3], [2, 3], [0, 2], [4, 5], [5, 7], [6, 7], [4, 6], [0, 4], [1, 5], [2, 6], [3, 7]]
    handle = ax.plot(fbox_vertices[lines_idx_list[0], 0], fbox_vertices[lines_idx_list[0], 1], fbox_vertices[lines_idx_list[0], 2])
    color = handle[0]._color
    for idxs in lines_idx_list:
        ax.plot(fbox_vertices[idxs, 0], fbox_vertices[idxs, 1], fbox_vertices[idxs, 2], color = color)

if __name__ == '__main__':
    franka = FrankaRobot()

    franka.set_base_offset([0, 0, 0.51])
    # franka.set_base_offset([0, 0, 0, 0, 0, 0, 0])

    init_joint = franka.INIT_JOINTS
    joint_target = np.array([0.0, 5e-1, 0.0, -2.3, 0.0, 2.8, 7.8e-1])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ee = franka.ee(init_joint)
    ee_pos = ee[:3]
    ee_q = quaternion.from_euler_angles(*ee[-3:])

    # draw franka
    franka_box_poses = franka.get_collision_boxes_poses(init_joint)
    for i, franka_box_pose in enumerate(franka_box_poses):
        fbox_pos = franka_box_pose[:3, 3]
        # print(fbox_pos)
        fbox_axes = franka_box_pose[:3, :3]

        fbox_vertex_offsets = franka._collision_box_vertices_offset[i]
        fbox_vertices = fbox_vertex_offsets.dot(fbox_axes.T) + fbox_pos
        # print(fbox_vertices)
        ax.scatter(fbox_vertices[:, 0], fbox_vertices[:, 1], fbox_vertices[:, 2])
        draw_box(ax, fbox_vertices)

    # draw tree
    tree_XYZ = np.array([[0.007936103414836028, 0.004735574035843012, 0.2598291172510352], [-0.003888823710599296, -0.0016676878913269412, 0.6723209614391856], [0.03160591448713911, -0.018215728283775788, 0.60618157012581], [-0.07664171565348726, 0.008513355122079163, 0.5057338636546936], [-0.02215363892535585, -0.0025054441122386835, 0.9401786700280585], [-0.08347075792256294, 0.009189168753933663, 0.8282315312103508], [0.04399492545194297, 0.032784020784865295, 1.0152665635185256], [-0.02404537537519225, 0.027012220333309293, 1.1101160027343098], [-0.08395985359689218, -0.17435413597403537, 0.646416060477702], [-0.025283628946139197, 0.12410467340648353, 0.756206914800063]])
    tree_RPY = np.array([[-0.03697488,  0.        , -1.03279744], [-0.03290285,  0.        , -1.32837932], [-0.42398421,  0.        , -1.92974421],[-1.63164481,  0.        ,  1.40271545],[-0.06551561,  0.        ,  0.81146456],[-1.66632523,  0.        ,  1.32763327],[-1.97033251,  0.        , -1.58815871],[-0.1223793 ,  0.        ,  2.53165676],[-1.17469293,  0.        ,  1.57922654],[-1.73461841e+00,  2.22044605e-16, -1.01528058e+00]])
    tree_LWH = np.array([[0.02539842, 0.02539842, 0.5       ],[0.02015874, 0.02015874, 0.325     ],[0.016  , 0.016  , 0.21125],[0.016    , 0.016    , 0.1373125],[0.016  , 0.016  , 0.21125],[0.016    , 0.016    , 0.1373125],[0.016    , 0.016    , 0.1373125],[0.016    , 0.016    , 0.1373125],[0.016     , 0.016     , 0.08925313],[0.016     , 0.016     , 0.08925313]])
    tree_stiffness = np.array([122.55971286623885, 122.55971286623885, 48.637854283196994, 48.637854283196994, 48.637854283196994, 19.30194526365572, 19.30194526365572, 19.30194526365572, 19.301945263655693, 19.301945263655693])

    boxes_vertices = []
    # for i in range(len(tree_XYZ)):
    #     box_pos, box_rpy, box_sizes = tree_XYZ[i], tree_RPY[i], tree_LWH[i]
    #     box = np.concatenate([box_pos, box_rpy, box_sizes])
    #     box_vertices = franka.get_obstacle_vertices(box)
    #     boxes_vertices.append(box_vertices)
    
    box = np.array([0.35, 0, 0.7, 0, 0, 0, 0.05, 0.1, 0.2])
    box_vertices = franka.get_obstacle_vertices(box)
    boxes_vertices.append(box_vertices)
    box = np.array([0.55, 0, 0.5 + 0.05 / 2 + 0.1, 0, 0, 0, 0.05, 0.05, 0.05])
    box_vertices = franka.get_obstacle_vertices(box)
    boxes_vertices.append(box_vertices)
    
    for i, box_vertices in enumerate(boxes_vertices):
        ax.scatter(box_vertices[:, 0], box_vertices[:, 1], box_vertices[:, 2])
        draw_box(ax, box_vertices)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.axis('off')

    plt.show()
