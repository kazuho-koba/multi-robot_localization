#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

# 一般的なパッケージの読み込み
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use('nbagg')   #　これはjupyter notebookとかでアニメーションを表示するためのもので、通常pythonだと逆に表示されなくなる
import matplotlib.animation as anm
import math
import random
import numpy as np
from scipy.stats import expon, norm, uniform

# クラスとして自分で定義した諸々の読み込み
from robot import *

# このファイルを直接実行した場合はここからスタートする
if __name__ == '__main__':

    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600.0                   # フィールド1辺長さ[m]
    SIM_TIME = 500                  # シミュレーション総時間 [sec]
    SAVE_VIDEO = True               # 動画ファイルを保存
    VIDEO_PLAY_SPEED = 10           # 動画ファイルの再生速度倍率
    ################################

    ################################
    # テスト実行
    ################################
    # 環境をオブジェクト化
    world = World(SIM_TIME, 1, debug=False,
                  field=FIELD,
                  save_video=SAVE_VIDEO, video_speed=VIDEO_PLAY_SPEED)

    # ランドマークを生成、地図に登録、地図と環境を紐付け
    m = Map()
    m.append_landmark(Landmark(100, 0))
    m.append_landmark(Landmark(0, 100))
    world.append(m)

    # エージェントを定義
    # straight = Agent(1.5, 0.0)
    # circling = Agent(1.5, 10.0/180*math.pi)
    # bs_agent = Agent(0.0, 0.0)

    # ロボットのオブジェクト化
    robots = [Robot(id=i, role='explorer',
                    pose=np.array([random.uniform(0, 100),
                                   random.uniform(0, 100),
                                   random.uniform(-math.pi, math.pi)]).T,
                    max_vel=MAX_VEL, field=FIELD)
              for i in range(NUM_BOTS)]

    # エージェント（コイツがロボットの動きを決める）のオブジェクト化
    agents = [Agent(id=i, nu=0, omega=0) for i in range(NUM_BOTS)]

    # すべてのロボットを環境に登録する
    for i in range(NUM_BOTS):

        # 基地局の設定
        if i == 0:
            robots[i].role = 'basestation'
            robots[i].pose = np.array([0, 0, 45.0/180*math.pi]).T

        # 各ロボットにエージェントとセンサを搭載
        robots[i].agent = agents[i]
        robots[i].sensor = Camera(m, robots[i], robots, field=FIELD)

        world.append(robots[i])
        # print(robots[i])

    world.draw()