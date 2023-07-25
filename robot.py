#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

# 一般的なパッケージの読み込み
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use('nbagg')   #　これはjupyter notebookとかでアニメーションを表示するためのもので、通常pythonだと逆に表示されなくなる
import matplotlib.animation as anm
import math, random
import numpy as np
from scipy.stats import expon, norm, uniform

# クラスとして自分で定義した諸々の読み込み
from ideal_robot import *

# より現実的なロボットのクラスをIdealRobotを継承して実装する
class Robot(IdealRobot):
    def __init__(self, id, role, pose,
                 max_vel, field,
                 agent = None, sensor = None,
                 color='black',
                 noise_per_meter=1, noise_std=math.pi/60,   # 1mあたりに生じるノイズの回数、ロボの向きに乗るノイズの標準偏差
                 bias_rate_stds=(0.1, 0.1),                 # 移動量に対するロボット固有のバイアス（前進、旋回）
                 ex_stuck_time=1000, ex_escape_time=10,     # スタック発生間隔の期待値、スタック脱出所要時間
                 ex_kidnap_time=3600*24                     # 誘拐の発生間隔の期待値
                 ):
        # 継承したIdealRobotクラスのinitを実行する
        super().__init__(id, role, pose, max_vel, field, agent, sensor, color)
        
        # ロボットの移動に加わるバイアスを定義
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))        # noise_per_meterは1mあたりのノイズ（踏みつける小石のイメージ）の個数平均値。その逆数はノイズを1つ引き当てる(=小石を踏む)までの前進距離
        self.distance_until_noise = self.noise_pdf.rvs()                    # 定義した確率分布から1つ値をドローしてそれを次の小石までの距離とする
        self.theta_noise = norm(scale=noise_std)                            # ロボの向きθに加える雑音を決めるガウス分布のオブジェクト
        
        # ロボットの1ステップあたり移動量に加わるバイアスを定義
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])      # 平均loc、標準偏差scaleのガウス分布から1つ値をドローしてこのロボット固有のバイアスにする
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        # ロボットのスタックに関する情報を定義
        self.stuck_pdf = expon(scale=ex_stuck_time)         # スタックが起こる確率分布（指数分布）
        self.escape_pdf = expon(scale=ex_escape_time)       # スタックから脱出する確率分布（指数分布
        self.time_until_stuck = self.stuck_pdf.rvs()        # 確率分布から次のスタックまでの時間をドロー
        self.time_until_escape = self.escape_pdf.rvs()      # 確率分布から上記スタックを脱出するのに要する時間をドロー
        self.is_stuck = False                               # 今まさにスタックしているかどうかを判別するフラグ

        # ロボットの誘拐に関する情報を定義
        kidnap_range_x, kidnap_range_y = (0, self.field), (0, self.field)   # 誘拐発生後の位置としてとりうる値の範囲
        self.kidnap_pdf = expon(scale=ex_kidnap_time)                       # 誘拐発生までの時間の確率分布
        self.time_until_kidnap = self.kidnap_pdf.rvs()                      # 誘拐発生までの時間をドロー
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0),
                                   scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi)) # 誘拐発生後の位置を与える確率分布
        

    # 毎秒呼ばれて今時タイムステップにノイズが乗るか（＝小石を踏むか）を判別するメソッド
    def noise(self, pose, nu, omega, time_interval):
        # 進んだ距離を小石までの距離から引いて、小石までの距離が0以下になったら踏んだと判定
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            # 踏んだ場合、次の小石までの距離をドローし直すとともに、ロボットの向きに雑音を加える
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
        
        return pose
    
    # ロボットの1ステップ移動量に雑音を加える処理
    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega
    
    # ロボットのスタックに関する処理
    def stuck(self, nu, omega, time_interval):
        # いま現にスタックしている場合
        if self.is_stuck:
            self.time_until_escape -= time_interval # 脱出所要時間としてセットされた時間が経過したかを監視

            # 経過した場合
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs() # 次のスタックで要する脱出時間を再度ドロー
                self.is_stuck = False                           # 脱出したのでスタックフラグをオフに

        # スタックしていない場合
        else:
            self.time_until_stuck -= time_interval  # 次のスタック時刻に達しているかをチェック

            # 達した場合はスタックさせる
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()   # 次のスタックが発生するまでの時間を再度ドロー（実際は脱出してから計時を始める
                self.is_stuck = True                            # 今まさにスタックが始まったのでフラグをオンに

        # 返り値
        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    # ロボットの誘拐に関する処理
    def kidnap(self, pose, time_interval):

        # 例外扱いしないと基地局も誘拐されてしまう
        if self.role != 'basestation':
            self.time_until_kidnap -= time_interval     # 誘拐発生までの時間の経過をモニタ
            # 時間が経過した場合、誘拐発生
            if self.time_until_kidnap <= 0.0:
                self.time_until_kidnap += self.kidnap_pdf.rvs() # 次の誘拐発生までの時間をドロー
                return np.array(self.kidnap_dist.rvs())         # 誘拐後の位置をドローして返り値とする
            
            else:
                return pose
        else:
            return pose

    # ロボットについて今時タイムステップに実行する処理
    def one_step(self, time_interval):
        if not self.agent:
            return
        
        obs = self.sensor.data(self.pose) if self.sensor else None
        
        # ロボットの1ステップあたり移動量を決める
        nu, omega, self.goal = self.agent.decision(self.pose, self.role, self.max_vel, self.current_time, obs)
        
        # 決定した移動量に従ってロボットの情報を更新
        nu, omega = self.bias(nu, omega)                                        # 移動量にバイアスを印加
        nu, omega = self.stuck(nu, omega, time_interval)                        # スタック判定とそれに応じた移動量の操作
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)  # 移動の実行
        self.pose = self.noise(self.pose, nu, omega, time_interval)             # 移動結果にノイズを加える
        self.pose = self.kidnap(self.pose, time_interval)                       # 誘拐に関する判定・処理
        self.current_time += time_interval

# このファイルを直接実行した場合はここからスタートする
if __name__=='__main__':

    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600                     # フィールド1辺長さ[m]
    SIM_TIME = 1000                  # シミュレーション総時間 [sec]
    SAVE_VIDEO = False              # 動画ファイルを保存
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
    robots = [Robot(id=i, role = 'explorer',
                    pose=np.array([random.uniform(0,100),
                                   random.uniform(0,100),
                                   random.uniform(-math.pi,math.pi)]).T,
                    max_vel = MAX_VEL, field = FIELD)
                    for i in range (NUM_BOTS)]
    
    # エージェント（コイツがロボットの動きを決める）のオブジェクト化
    agents = [Agent(id=i, nu=0, omega=0) for i in range (NUM_BOTS)]

    # すべてのロボットを環境に登録する
    for i in range(NUM_BOTS):
                
        # 基地局の設定
        if i == 0:
            robots[i].role = 'basestation'
            robots[i].pose = np.array([0,0,45.0/180*math.pi]).T
        
        # 各ロボットにエージェントとセンサを搭載
        robots[i].agent = agents[i]
        robots[i].sensor = IdealCamera(m, robots[i], robots)

        world.append(robots[i])
        # print(robots[i])


    world.draw()
