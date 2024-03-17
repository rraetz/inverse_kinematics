# Copyright (c) 2024 rraetz

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np


def generate_random_joint_positions(q_min: np.array, q_max: np.array) -> np.array:
    return np.random.uniform(q_min, q_max)


def generate_random_pose(robot: rtb.Robot, q_min: np.array, q_max: np.array) -> sm.SE3:
    q = generate_random_joint_positions(q_min, q_max)
    pose = robot.fkine(q, end=robot.ee_links[0])
    return pose
