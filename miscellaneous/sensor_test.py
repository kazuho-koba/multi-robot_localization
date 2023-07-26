import sys, os, copy, math
import pandas as pd
abs_module_path = '/home/yal/kazuho/multi-robot_localization'
sys.path.append(abs_module_path)
from robot import *

m = Map()
m.append_landmark(Landmark(50, 0))  # ランドマークを（50, 0)に置く

robots = [Robot(id=i, role='explorer',
                pose=np.array([600,600,0]).T,
                max_vel=np.array([2.0, 1.0]), field=600)
            for i in range(2)]

distance = []
direction = []

# 1000回以下の処理を実行
for i in range(1000):
    c = Camera(m, myself=robots[0], robots=robots, field=600)                               # カメラを生成する
    d = c.data(np.array([0.0, 0.0, 0.0]).T)    # カメラの位置を原点、右向きにセット
    if len(d) > 0:
        distance.append(d[0][0][0])
        direction.append(d[0][0][1])

df = pd.DataFrame()
df['distance'] = distance
df['direction'] = direction
print(df)
print(df.std())
print(df.mean())