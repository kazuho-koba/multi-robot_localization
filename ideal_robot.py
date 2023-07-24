#!/usr/bin/env python
# -*- coding: utf-8 -*-
# code for python3

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# matplotlib.use('nbagg')   #　これはjupyter notebookとかでアニメーションを表示するためのもので、通常pythonだと逆に表示されなくなる
import matplotlib.animation as anm
import math, random
import numpy as np

# 環境を定義するクラス
class World:
    def __init__(self, time_span, time_interval, debug=False,
                 save_video = False, video_speed = 10):
        self.objects = []                   # ロボット、ランドマーク等のあらゆるオブジェクトがここに格納される
        self.debug = debug                  # Trueのときはアニメーションを切ってデバッグしやすくする
        self.time_span = time_span          # シミュレーション時間
        self.time_interval= time_interval   # タイムステップ幅
        self.save_video = save_video        # 動画を保存するかどうか
        self.video_speed = video_speed      # 動画の再生速度（×倍速）

    # オブジェクトを世界の一部として登録するためのメソッド
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(10, 10))      # 8*8 inch の図を準備
        ax = fig.add_subplot(111)               # サブプロットを準備
        ax.set_aspect('equal')                  # 縦横比を座標の値と一致させる
        ax.set_xlim(-50, 650)                     # x軸の範囲
        ax.set_ylim(-50, 650)                     # y軸の範囲
        ax.set_xlabel("X", fontsize = 20)       # x軸ラベル
        ax.set_ylabel("y", fontsize = 20)       # y軸ラベル

        # アニメーション関連の処理
        elems = []                              # 描画する図形のリスト
        if self.debug:
            for i in range(100):
                self.one_step(i, elems, ax)     # デバッグ時はアニメーションさせない
        else:
            # アニメーションするオブジェクトを作る（intervalは更新周期[ms]
            self.max_anm_speed_scale = 10
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                         frames=int(self.time_span/self.time_interval),
                                         interval=int(self.time_interval*1000/self.max_anm_speed_scale), repeat=False)
            # アニメーションの保存
            if self.save_video == True:
                self.ani.save('animation.mp4', writer='ffmpeg', fps=self.video_speed)
            plt.show()
            pass


    def one_step(self, i, elems, ax):
        while elems:
            elems.pop().remove()
        
        # 時刻の描画
        elems.append(ax.text(-25,625, "t= %.1f[s]" %(self.time_interval*i+1), fontsize=10))
        
        # 他のオブジェクトの描画
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)
        
        pass
        
# 理想ロボット（動作について誤差などが発生しない）を定義するクラス
class IdealRobot:
    def __init__(self, id, role, pose,
                 agent = None, sensor = None,
                 color='black'):
        self.id = id            # ロボットのID
        self.role = role        # ロボットの役割
        self.pose = pose        # ロボットの位置・姿勢の初期値
        self.agent = agent      # ロボットを動かす主体（コイツが速度指令値とか決める）
        self.sensor = sensor    # ロボットに搭載されているセンサ
        self.color = color      # ロボットの描画上の色
        self.r = 5              # （描画上の）ロボットの半径
        self.pose_log = [pose]  # 軌跡を描画するために今までの位置と姿勢を格納する
        self.current_time = 0   # 現在時刻

    # ロボットの状態遷移（速度指令に基づく位置の更新）を実行するメソッド
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        
        # 角速度が小さい時とそれ以外で場合分け
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0), nu*math.sin(t0), omega])*time
        else:
            return pose + np.array([nu/omega*(math.sin(t0 + omega*time)-math.sin(t0)),
                                    nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                    omega*time])
        
    # ロボットがエージェントの決定に従って速度指令を受け取るメソッド
    def one_step(self, time_interval):
        # エージェントがロボットに搭載されていなければ何もしない
        if not self.agent:
            return
        
        if self.sensor:
            obs = self.sensor.data(self.pose)
        else:
            None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.current_time += time_interval
        # print(self.current_time)

    # ロボットを描画するメソッド
    def draw(self, ax, elems):
        # 描画時のロボットの色を設定
        if self.role == 'basestation':
            self.color = 'black'
        elif self.role == 'repeater':
            self.color = 'green'
        elif self.role == 'groupleader':
            self.color = 'red'
        else:
            self.color = 'blue'

        # それ以外の描画に必要な情報を設定
        x, y, theta = self.pose                             # 位置・姿勢を示すposeを改めて3つの変数に格納
        xn = x + self.r*math.cos(theta)                     # ロボットの鼻先のx座標
        yn = y + self.r*math.sin(theta)                     # 同じくy座標
        elems += ax.plot([x, xn], [y, yn], color=self.color)# ロボットの向き（姿勢）を表す線分の描画
        c = patches.Circle(xy=(x, y), radius=self.r,
                           fill=False, color=self.color)    # ロボット本体を示す円の設定
        elems.append(ax.add_patch(c))                       # ロボット本体を示す円を描画
        self.pose_log.append(self.pose)                     # ロボットの位置・姿勢ログに現在のそれを追加
        
        # ロボット軌跡の描画
        if self.current_time == 0:
            del self.pose_log[0]                                # ログの0番目は削除
        elems += ax.plot([e[0] for e in self.pose_log[-20:]],
                         [e[1] for e in self.pose_log[-20:]],
                         linewidth = 0.5, color='black')        # ロボットの軌跡（直近20タイムステップ分）を描画リストに追加

        # センサによる検知の様子の描画
        if self.sensor and len(self.pose_log) > 1:
            self.sensor.draw(ax, elems, self.pose_log[-2])
        # エージェントの情報を描画
        if self.agent and hasattr(self.agent, 'draw'):
            self.agent.draw(ax, elems)

# ロボットを動かすエージェントのクラス
class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega
    
# ランドマークのクラス（今回は使わないかも）
class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker='*',
                       label='landmarks', color='orange')
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], 'id: '+str(self.id), fontsize=10))

# 地図のクラス（環境Worldにオーバーレイされるイメージ。ランドマークなどが登録される）
class Map:
    def __init__(self):
        self.landmarks = []     # 空のランドマークのリストを準備する

    # ランドマークを追加するメソッド
    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks) + 1
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)

# 理想センサ（カメラ、ただし観測誤差が生じない）のクラス
class IdealCamera:
    def __init__(self, env_map,
                 distance_range=(1.0, 90.0), direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []                      # 各対象について最後に計測したときの情報
        self.distance_range = distance_range    # 計測可能距離レンジ
        self.direction_range = direction_range  # 計測可能角度レンジ

    # ランドマークが計測できるかどうかを判別するメソッド
    def visible(self, polarpos):
        if polarpos is None:
            return False
        
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]\
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            if self.visible(p):
                observed.append((p, lm.id))

        self.lastdata = observed
        return observed
    
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]

        # 対象が見えている方位角を-pi〜piの範囲に収める
        while phi >= np.pi:
            phi -= 2*np.pi
        while phi < -np.pi:
            phi += 2*np.pi

        # 返り値。hypotは2乗和の平方根を返す。*diffはdiffの要素それぞれ
        return np.array([np.hypot(*diff), phi]).T
    
    # 計測の様子を描画
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance*math.cos(direction+theta)
            ly = y + distance*math.sin(direction+theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")

# このファイルを直接実行した場合はここからスタートする
if __name__=='__main__':
    ################################
    # シミュレーションの設定
    NUM_BOTS = 3            # ロボット総数
    SAVE_VIDEO = True       # 動画ファイルを保存
    VIDEO_PLAY_SPEED = 10   # 動画ファイルの再生速度倍率
    ################################

    ################################
    # テスト実行用
    ################################
    # 環境をオブジェクト化
    world = World(50, 1, debug=False,
                save_video=SAVE_VIDEO, video_speed=VIDEO_PLAY_SPEED)     

    # ランドマークを生成、地図に登録、地図と環境を紐付け
    m = Map()
    m.append_landmark(Landmark(100, 0))
    m.append_landmark(Landmark(0, 100))
    world.append(m)

    # エージェントを定義
    straight = Agent(1.5, 0.0)
    circling = Agent(1.5, 10.0/180*math.pi)
    bs_agent = Agent(0.0, 0.0)

    # ロボットのオブジェクト化
    robots = [IdealRobot(id=i, role = 'explorer',
                        pose=np.array([random.uniform(0,100),
                                        random.uniform(0,100),
                                        random.uniform(-math.pi,math.pi)]).T,
                        sensor = IdealCamera(m))
                        for i in range (NUM_BOTS)]

    # すべてのロボットを環境に登録する
    for i in range(NUM_BOTS):
        # 基地局の設定
        if i == 0:
            robots[i].role = 'basestation'
            robots[i].pose = np.array([0,0,0]).T
            robots[i].agent = bs_agent
        else:
            robots[i].agent = circling
        world.append(robots[i])
        # print(robots[i])


    world.draw()
