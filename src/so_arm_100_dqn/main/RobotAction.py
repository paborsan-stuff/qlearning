"""
Educational Use License

Copyright (c) 2025 UP Student IA Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, and distribute the Software solely for educational and
non-commercial purposes, subject to the following conditions:

[Include the full text as above or a reference to the LICENSE file]

This license is intended for educational purposes only.
"""

import numpy as np
import mujoco

class RobotAction:
    def __init__(self, mdl):
        self.mdl = mdl

    def generate_action(self):
        """
        Generate a random action within the joint limits.
        This action can be interpreted as the desired change in joint angles.
        """
        actions = []
        joint_limits = []
        for i in range(self.mdl.njnt):
            joint_name = mujoco.mj_id2name(self.mdl, mujoco.mjtObj.mjOBJ_JOINT, i)
            lower, upper = self.mdl.jnt_range[i]
            print(f"Joint: {joint_name}, Range: {lower} to {upper}")
            joint_limits.append((lower, upper))

            # Generate a random value within the joint's limits
            random_val = np.random.uniform(lower, upper)
            actions.append(random_val)

        print("Joint limits:", joint_limits)
        return np.array(actions)