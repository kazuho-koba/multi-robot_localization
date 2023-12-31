import sys, os, copy, math
import pandas as pd
abs_module_path = '/home/yal/kazuho/multi-robot_localization'
sys.path.append(abs_module_path)
from robot import *



world = World(30, 1)

initial_pose = np.array([0, 0, 0]).T
robots = []

for i in range(100):
    r = Robot(1, 1, initial_pose, np.array([2.0, 1.0]), 600, agent=Agent(1, 1, 0.0, math.pi*10.0/180), sensor = None)
    world.append(r)
    robots.append(r)

world.draw()

poses = pd.DataFrame([[math.sqrt(r.pose[0]**2 + r.pose[1]**2), r.pose[2]] for r in robots],
                     columns=['r', 'theta'])
print(poses.transpose())
print(poses['theta'].var())
print(poses['theta'].mean())
print(math.sqrt(poses['theta'].var()/poses['theta'].mean()))