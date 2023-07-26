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
from ideal_robot import *

# より現実的なロボットのクラスをIdealRobotを継承して実装する
class Robot(IdealRobot):
    def __init__(self, id, role, pose,
                 max_vel, field,
                 agent=None, sensor=None,
                 color='black',
                 noise_per_meter=0.1, noise_std=math.pi/180,   # 1mあたりに生じるノイズの回数、ロボの向きに乗るノイズの標準偏差
                 # 移動量に対するロボット固有のバイアス（前進、旋回）
                 bias_rate_stds=(0.05, 0.025),
                 ex_stuck_time=1000, ex_escape_time=10,     # スタック発生間隔の期待値、スタック脱出所要時間
                 ex_kidnap_time=3600*24                     # 誘拐の発生間隔の期待値
                 ):
        # 継承したIdealRobotクラスのinitを実行する
        super().__init__(id, role, pose, max_vel, field, agent, sensor, color)

        # 基地局から通知された情報
        self.informed_pose = []
        self.informed_time = 0

        # ロボットの移動に加わるバイアスを定義
        # noise_per_meterは1mあたりのノイズ（踏みつける小石のイメージ）の個数平均値。その逆数はノイズを1つ引き当てる(=小石を踏む)までの前進距離
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        # 定義した確率分布から1つ値をドローしてそれを次の小石までの距離とする
        self.distance_until_noise = self.noise_pdf.rvs()
        # ロボの向きθに加える雑音を決めるガウス分布のオブジェクト
        self.theta_noise = norm(scale=noise_std)

        # ロボットの1ステップあたり移動量に加わるバイアスを定義
        # 平均loc、標準偏差scaleのガウス分布から1つ値をドローしてこのロボット固有のバイアスにする
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        # ロボットのスタックに関する情報を定義
        # スタックが起こる確率分布（指数分布）
        self.stuck_pdf = expon(scale=ex_stuck_time)
        # スタックから脱出する確率分布（指数分布
        self.escape_pdf = expon(scale=ex_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()        # 確率分布から次のスタックまでの時間をドロー
        self.time_until_escape = self.escape_pdf.rvs()      # 確率分布から上記スタックを脱出するのに要する時間をドロー
        self.is_stuck = False                               # 今まさにスタックしているかどうかを判別するフラグ

        # ロボットの誘拐に関する情報を定義
        kidnap_range_x, kidnap_range_y = (
            0, self.field), (0, self.field)   # 誘拐発生後の位置としてとりうる値の範囲
        # 誘拐発生までの時間の確率分布
        self.kidnap_pdf = expon(scale=ex_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()                      # 誘拐発生までの時間をドロー
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0),
                                   scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi))  # 誘拐発生後の位置を与える確率分布

    # 毎秒呼ばれて今時タイムステップにノイズが乗るか（＝小石を踏むか）を判別するメソッド
    def noise(self, pose, nu, omega, time_interval):
        # 進んだ距離を小石までの距離から引いて、小石までの距離が0以下になったら踏んだと判定
        self.distance_until_noise -= abs(nu)*time_interval + \
            self.r*abs(omega)*time_interval
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
            self.time_until_escape -= time_interval  # 脱出所要時間としてセットされた時間が経過したかを監視

            # 経過した場合
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()  # 次のスタックで要する脱出時間を再度ドロー
                self.is_stuck = False                           # 脱出したのでスタックフラグをオフに

        # スタックしていない場合
        else:
            self.time_until_stuck -= time_interval  # 次のスタック時刻に達しているかをチェック

            # 達した場合はスタックさせる
            if self.time_until_stuck <= 0.0:
                # 次のスタックが発生するまでの時間を再度ドロー（実際は脱出してから計時を始める
                self.time_until_stuck += self.stuck_pdf.rvs()
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
                self.time_until_kidnap += self.kidnap_pdf.rvs()  # 次の誘拐発生までの時間をドロー
                # 誘拐後の位置をドローして返り値とする
                return np.array(self.kidnap_dist.rvs())

            else:
                return pose
        else:
            return pose

    # ロボットについて今時タイムステップに実行する処理
    def one_step(self, time_interval):
        # エージェントが搭載されていない場合、何もしない
        if not self.agent:
            return
        
        # print(self.id, self.informed_pose, self.informed_time)
        if self.pose[2] > math.pi or self.pose[2] < -math.pi:
            print(self.id, self.pose)

        # センサ情報を取得
        if self.sensor:
            self.obs = self.sensor.data(self.pose)
            # print(self.obs)
        else:
            self.obs = None


        # ロボットの1ステップあたり移動量を決める（返り値にはこのあとノイズなどが乗る）
        nu, omega, self.goal = self.agent.move_to_goal(
            self.pose, self.role, self.max_vel, self.current_time, self.obs)
        
        # その他の意思決定をエージェントから受け取る（現時点では何もしない、継承先でdecisionメソッドごと上書き）
        nu, omega = self.agent.decision(self.obs)

        # 基地局エージェントによる他ロボット観測結果の処理
        if self.role == 'basestation':
            self.agent.bs_task()

        # 決定した移動量に従ってロボットの情報を更新
        # 移動量にバイアスを印加
        nu, omega = self.bias(nu, omega)
        # スタック判定とそれに応じた移動量の操作
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(
            nu, omega, time_interval, self.pose)  # 移動の実行
        self.pose = self.noise(self.pose, nu, omega,
                               time_interval)             # 移動結果にノイズを加える
        # 誘拐に関する判定・処理
        self.pose = self.kidnap(self.pose, time_interval)
        self.current_time += time_interval

# より現実的なカメラのクラスをIdealCameraクラスを継承して実装する
class Camera(IdealCamera):
    def __init__(self, env_map, myself, robots, field,
                 distance_range=(1.0, 90.0), direction_range=(-math.pi, math.pi),
                 distance_noise_rate=0.1, direction_noise=math.pi/90,               # センサへのノイズ
                 distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi/90,   # センサのバイアス
                 phantom_prob=0.01,                                                 # センサが幻影をみる確率
                 oversight_prob=0.05,                                               # センサが見落としをする確率
                 occulusion_prob = 0.01                                             # センサがオクルージョンを起こす確率
                 ):
        # 継承元であるCameraクラスの初期化プロセスを呼び出し
        super().__init__(env_map, myself, robots, field, distance_range, direction_range)

        # センサ値に加わるノイズの設定
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        # センサ値に加わるバイアスの設定
        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev) # センサごとに固有のバイアスを初期化時に設定
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)             # 同上

        # センサが幻影を見る確率に関する設定
        phantom_range_x, phantom_range_y = (0.0, field), (0.0, field)
        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0], 0), scale=(rx[1]-rx[0], ry[1]-ry[0], 0)) # 幻影が生じうる場所の分布
        self.phantom_prob = phantom_prob                                                        # 幻影が発生する確率

        # センサの見落としに関する設定
        self.oversight_prob = oversight_prob

        # センサのオクルージョンに関する設定
        self.occulusion_prob = occulusion_prob


    # 計測値にノイズを加えて返すメソッド
    def noise(self, relpos):
        # 計測距離を真値周辺のガウス分布からドロー
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        # カメラから対象への視線角を（以下同文
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        # 対象自体の空間角を（以下同文
        theta = norm.rvs(loc=relpos[2], scale=self.direction_noise)
        
        return np.array([ell, phi, theta]).T
    
    # 計測値にバイアスを加えて返すメソッド
    def bias(self, relpos):
        return relpos + np.array([relpos[0]*self.distance_bias_rate_std,
                                  self.direction_bias, 0]).T
    
    # 一定確率で幻影を生じさせるメソッド
    # これはセンサの覆域内かどうかの判定より前に呼ばれるので、覆域内に何もなくても幻影が生じうる
    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos
        
    # センサの見落としを加味するメソッド
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos
        
    # センサのオクルージョンに関する処理（ここでは、観測対象とセンサの間に何かがかぶるというよりは、
    # 本来より近くにあるように誤認するという形で単純化する
    def occulusion(self, relpos):
        if uniform.rvs() < self.occulusion_prob:
            ell = relpos[0] + uniform.rvs()*(self.distance_range[1] - relpos[0])
            return np.array([ell, relpos[1], relpos[2]]).T
        else:
            return relpos

    # 環境中のオブジェクトを実際に計測する（そおそもセンサレンジ内かどうかも判定する
    def data(self, cam_pose):
        # センサで観測できたオブジェクトのリスト
        observed = []

        # ランドマークの計測
        for lm in self.map.landmarks:
            # センサからの見え方を演算
            z = self.observation_function(cam_pose, lm.pose)# ランドマークを計測
            z = self.phantom(cam_pose, z)                   # 幻影を見てランドマークの位置を大きく誤計測する可能性を扱う処理
            z = self.occulusion(z)                          # オクルージョンの可能性を考慮
            z = self.oversight(z)                           # 見落としの可能性を処理
                        
            # センサの範囲内であればノイズなどを加えて観測リストに追加
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append(('landmark', z, lm.id))


        # 他のロボットの計測
        for bot in self.robots:
            if bot.id != self.myself.id:    # 自分自身は計測しない
                # センサからの見え方を演算
                sensed_bot = self.observation_function(cam_pose, bot.pose)  # 他のロボットの位置を計測
                sensed_bot = self.phantom(cam_pose, sensed_bot)             # 幻影を見て計測した他ロボの位置を大きくご計測する可能性を扱う処理
                sensed_bot = self.occulusion(sensed_bot)                    # オクルージョンの可能性を考慮
                sensed_bot = self.oversight(sensed_bot)                     # 見落としの確率を処理

                # センサの範囲内であればノイズなどを加えて観測リストに追加
                if self.visible(sensed_bot):
                    sensed_bot = self.bias(sensed_bot)
                    sensed_bot = self.noise(sensed_bot)
                    observed.append(('robot', sensed_bot, bot.id, bot.role))
        
        self.lastdata = observed
        return observed


# このファイルを直接実行した場合はここからスタートする
if __name__ == '__main__':

    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600.0                   # フィールド1辺長さ[m]
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
    m.append_landmark(Landmark(100, 0, 0))
    m.append_landmark(Landmark(0, 100, 0))
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
    # 基地局は特殊なのでその設定を追加
    robots[0].role = 'basestation'
    robots[0].pose = np.array([0, 0, 0])

    # エージェント（コイツがロボットの動きを決める）のオブジェクト化
    agents = [Agent(id=robots[i].id, role=robots[i].role, nu=0, omega=0) for i in range(NUM_BOTS)]

    # すべてのロボットを環境に登録する
    for i in range(NUM_BOTS):

        # 各ロボットにエージェントとセンサを搭載
        robots[i].agent = agents[i]
        robots[i].sensor = Camera(m, robots[i], robots, field=FIELD)

        world.append(robots[i])

    world.draw()
