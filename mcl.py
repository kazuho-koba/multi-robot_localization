# パッケージのインポート
from scipy.stats import multivariate_normal
import copy, random

# 独自に定義したクラスファイルなどのインポート
from robot import *

# 各パーティクルを表現するクラスを作る
class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose   # パーティクルの位置・向き
        self.weight = weight    # パーティクルの重み（全パーティクルの和が1）

    # パーティクルの動きを実装するメソッド
    def motion_update(self, nu, omega, time, noise_rate_pdf):

        ns = noise_rate_pdf.rvs()   # ノイズをドロー（順にnn, no, on, oo）
        # ノイズが乗った移動量を演算
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        
        # 移動量に従ってパーティクルを動かす（パーティクルは仮想的な存在で小石を踏んだりしないので、IdealRobotクラスの移動メソッドが転用できる
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
    
    # 自己位置推定器から呼び出されてパーティクルから見たカメラの観測結果を返す
    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            # ランドマークを検知した場合
            if d[2] == 'landmark':
                obs_pos = d[0]
                obs_id = d[1]

                # パーティクルの位置と地図からランドマークの距離と方角を算出(自己位置の候補であり確定的な位置座標であるパーティクルの位置から
                # 既知のランドマークを見たらどう見えるか、という演算なので、CameraでなくIdealCameraの観測関数を呼び出している)
                pos_on_map = envmap.landmarks[obs_id].pos
                particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

                # 尤度の計算
                distance_dev = distance_dev_rate * particle_suggest_pos[0]
                cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
                self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)
            
            # 他のロボットを検知した場合
            elif d[2] == 'robot':
                # print(envmap.robots[d[1]-1])
                pass

# パーティクルを管理する自己位置情報推定器のクラスを作る
class Mcl:
    def __init__(self, envmap, init_pose, num,
                 motion_noise_stds={"nn":0.8,"no":0.001,"on":0.005,"oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05):
        # numで指定された数だけパーティクルのオブジェクトを用意する
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]

        # パーティクルごとの尤度を計算するための情報
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        # 尤度最大のパーティクルを出力するための情報
        self.ml = self.particles[0] # 尤度最大のパーティクルを格納する変数を1番目のパーティクルとして初期化
        self.pose = self.ml.pose    # 尤度最大のパーティクルの位置・姿勢

        v = motion_noise_stds                                           # パーティクルの位置に加わる、ガウス分布に従う雑音に関する標準偏差
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])   # 与えられたリストを対角成分に持つ対角行列を作る（この場合は4*4行列で、これが4次元ガウス分布の共分散行列になる？）
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)         # 4次元ガウス分布

    # 各パーティクルを動かすメソッド（Particleクラスのメソッドを呼び出し
    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    # エージェントから呼び出されてカメラ情報の処理をする
    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        
        # この時点で尤度最大のパーティクル情報を取得
        self.set_ml()

        # パーティクルのリサンプリングを実施する（何も観測していない時に実行しても
        # その時点の自分の認識のバイアスを肥大化させるだけなので、観測物が存在するときのみ実行する
        # if len(observation) > -1:
        #     self.resampling()   

        # 修正したリサンプリングでは上記の心配がない
        self.resampling_mod()

    # パーティクルのリサンプリングを実施するメソッド
    def resampling(self):
        ws = [e.weight for e in self.particles]     # パーティクルの重みリストを取得

        # 重みの和が小さすぎてゼロとして扱われるとエラーになるので小さい数を足す
        if sum(ws) < 1e-100:
            ws = [e + 1e-100 for e in ws]

        # wsに比例した確率でパーティクルをk個選ぶ
        ps = random.choices(self.particles, weights=ws, k=len(self.particles))  

        # 選んだパーティクルの重みを均一に
        self.particles = [copy.deepcopy(e) for e in ps] 
        for p in self.particles:
            p.weight = 1.0/len(self.particles)

    # リサンプリングの計算量とサンプリングバイアスを改良したもの
    def resampling_mod(self):
        ws = np.cumsum([e.weight for e in self.particles])  # その要素までの重みの累積和を得る
        
        # 重みの和が小さすぎてゼロとして扱われるとエラーになるので小さい数を足す
        if ws[-1] < 1e-100:
            ws = [e + 1e-100 for e in ws]

        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles:
            p.weight = 1.0/len(self.particles)
            
    # 尤度最大のパーティクルを取得するメソッド
    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose

    # パーティクルを描画するためのメソッド
    def draw(self, ax, elems):
        # 各パーティクルの情報をリストとして取得
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]

        # 描画リストに追加
        elems.append(ax.quiver(xs, ys, vxs, vys,
                               angles='xy', scale_units='xy', color='blue', alpha=0.5))

# Agentクラスを継承して自己位置推定を行うエージェントのクラスを作る
class EstimationAgent(Agent):
    def __init__(self, time_interval, id, nu, omega, robot=None, estimator=None):
        # 継承元のAgentクラスの初期化処理を実行
        super().__init__(id, nu, omega, robot)
        self.estimator = estimator
        self.time_interval = time_interval

        # ロボットに1ステップ前に指示した移動量（これにノイズを載せてパーティクルの移動量にする）
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    # エージェントが意思決定をするメソッド（継承元のメソッドを上書き
    def decision(self, observation=None):
        # パーティクルの動きを演算する
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        
        # 今時ステップにロボットに指示した移動量（コレ自体は継承元クラスで目標位置をベースに演算）を控えておく
        self.prev_nu, self.prev_omega = self.nu, self.omega

        # カメラ情報を取得
        self.estimator.observation_update(observation)
        
        return self.nu, self.omega
    
    # 描画に関する処理（エージェントが想定する自己位置に関する信念を描写する）
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

        # 尤度最大のパーティクルの姿勢を描画
        x,y,t = self.estimator.pose     # 座標の取得
        s = "({:.1f}, {:.1f}, {})".format(x, y, int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=10))
        elems.append(ax.quiver(x, y, math.cos(t), math.sin(t),
                               color='red', scale=15))


# このファイルを直接実行した場合はここからスタートする
if __name__=='__main__':
    
    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600                     # フィールド1辺長さ[m]
    SIM_TIME = 1000                  # シミュレーション総時間 [sec]
    TIME_STEP = 1                   # 1ステップあたり経過する秒数
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

    # ランドマークを生成、地図に登録
    m = Map()
    m.append_landmark(Landmark(100, 0))
    m.append_landmark(Landmark(0, 100))
    

    # ロボットのオブジェクト化
    robots = [Robot(id=i, role='explorer',
                    pose=np.array([random.uniform(0, 100),
                                   random.uniform(0, 100),
                                   random.uniform(-math.pi, math.pi)]).T,
                    max_vel=MAX_VEL, field=FIELD)
              for i in range(NUM_BOTS)]
    
    # 基地局は特殊なのでその設定を追加
    robots[1].role = 'basestation'
    robots[1].pose = np.array([0, 0, 45.0/180*math.pi])

    # エージェント（コイツがロボットの動きを決める）のオブジェクト化
    estimators = [Mcl(m, init_pose=robots[i].pose, num=100,
                      motion_noise_stds={"nn":0.8,"no":0.001,"on":0.005,"oo":0.2})
                  for i in range(NUM_BOTS)]                 # 各エージェントに搭載する自己位置推定器の定義
    agents = [EstimationAgent(time_interval=TIME_STEP, id=i, nu=0, omega=0, 
                              robot=robots[i], estimator=estimators[i])
              for i in range (NUM_BOTS)]                    # 各エージェントを定義

    # すべてのロボットにエージェントやセンサを搭載して、環境に登録する
    for i in range(NUM_BOTS):
        # 各ロボットにエージェントとセンサを搭載
        robots[i].agent = agents[i]
        robots[i].sensor = Camera(m, robots[i], robots, field=FIELD)

        # ロボットを環境に登録
        world.append(robots[i])
        
        # ロボットを地図に登録
        m.append_robot(robots[i])

    # 地図を環境に登録
    world.append(m)

    world.draw()