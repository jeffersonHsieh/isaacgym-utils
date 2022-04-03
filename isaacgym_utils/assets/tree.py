import numpy as np
from pathlib import Path

from isaacgym import gymapi
from isaacgym_utils.constants import isaacgym_utils_ASSETS_PATH
from isaacgym_utils.math_utils import transform_to_RigidTransform, vec3_to_np, quat_to_rot, np_to_vec3

from .assets import GymURDFAsset



class GymTree(GymURDFAsset):

    INIT_JOINTS = np.array([0, -np.pi / 4])
    _LOWER_LIMITS = None
    _UPPER_LIMITS = None
    _VEL_LIMITS = None

    _URDF_PATH = 'franka_description/robots/tree_test.urdf'

    @staticmethod
    def _key(env_idx, name):
        return (env_idx, name)

    def __init__(self, cfg, *args, actuation_mode='joints'):
        if 'urdf' in cfg:
            urdf_path = cfg['urdf']
            assets_root = Path(cfg['assets_root'])
        else:
            urdf_path = GymTree._URDF_PATH
            assets_root = isaacgym_utils_ASSETS_PATH

        super().__init__(urdf_path, *args,
                        shape_props=cfg['shape_props'],
                        dof_props=cfg['dof_props'],
                        asset_options=cfg['asset_options'],
                        assets_root=assets_root
                        )

    def get_link_transform(self, env_idx, name, link_name):
        transform = self.get_rb_transform(env_idx, name, link_name) 
        return transform

