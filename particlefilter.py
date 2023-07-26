# パッケージのインポート
from scipy.stats import multivariate_normal

# 独自に定義したクラスファイルなどのインポート
from robot import *

# 各パーティクルを表現するクラスを作る
class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

    # パーティクルの動きを実装するメソッド
    def motion_update(self, nu, omega, time, noise_rate_pdf):

        ns = noise_rate_pdf.rvs()   # ノイズをドロー（順にnn, no, on, oo）
        # ノイズが乗った移動量を演算
        noised_nu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        noised_omega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        
        # 移動量に従ってパーティクルを動かす（パーティクルは仮想的な存在で小石を踏んだりしないので、IdealRobotクラスの移動メソッドが転用できる
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)
    

# パーティクルを管理するクラスを作る
class Mcl:
    def __init__(self, init_pose, num, motion_noise_stds):
        # numで指定された数だけパーティクルのオブジェクトを用意する
        self.particles = [Particle(init_pose) for i in range(num)]

        v = motion_noise_stds                                           # パーティクルの位置に加わる、ガウス分布に従う雑音に関する標準偏差
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])   # 与えられたリストを対角成分に持つ対角行列を作る（この場合は4*4行列で、これが4次元ガウス分布の共分散行列になる？）
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)         # 4次元ガウス分布

    # 各パーティクルを動かすメソッド（Particleクラスのメソッドを呼び出し
    def motion_update(self, nu, omega, time):
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    # パーティクルを描画するためのメソッド
    def draw(self, ax, elems):
        # 各パーティクルの情報をリストとして取得
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]

        # 描画リストに追加
        elems.append(ax.quiver(xs, ys, vxs, vys, color='blue', alpha=0.5))

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
        
        return self.nu, self.omega
    
    # 描画に関する処理（エージェントが想定する自己位置に関する信念を描写する）
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)


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

    # ランドマークを生成、地図に登録、地図と環境を紐付け
    m = Map()
    m.append_landmark(Landmark(100, 0))
    m.append_landmark(Landmark(0, 100))
    world.append(m)


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
    estimators = [Mcl(init_pose=robots[i].pose, num=100,
                      motion_noise_stds={"nn":0.01,"no":0.02,"on":0.03,"oo":0.04})
                  for i in range(NUM_BOTS)]                 # 各エージェントに搭載する自己位置推定器の定義
    agents = [EstimationAgent(time_interval=TIME_STEP, id=i, nu=0, omega=0, 
                              robot=robots[i], estimator=estimators[i])
              for i in range (NUM_BOTS)]                    # 各エージェントを定義

    # すべてのロボットにエージェントやセンサを搭載して、環境に登録する
    for i in range(NUM_BOTS):
        # 各ロボットにエージェントとセンサを搭載
        robots[i].agent = agents[i]
        robots[i].sensor = Camera(m, robots[i], robots, field=FIELD)

        world.append(robots[i])
        # print(robots[i])

    world.draw()