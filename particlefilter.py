from robot import *

# 各パーティクルを表現するクラスを作る
class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

# パーティクルを管理するクラスを作る
class Mcl:
    def __init__(self, init_pose, num):
        # numで指定された数だけパーティクルのオブジェクトを用意する
        self.particles = [Particle(init_pose) for i in range(num)]

# Agentクラスを継承して自己位置推定を行うエージェントのクラスを作る
class EstimationAgent(Agent):
    def __init__(self, id, nu, omega, robot=None, estimator=None):
        # 継承元のAgentクラスの初期化処理を実行
        super().__init__(id, nu, omega, robot)
        self.estimator = estimator

    # 描画に関する処理（エージェントが想定する自己位置に関する信念を描写する）
    def draw(self, ax, elems):
        pass


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


    # ロボットのオブジェクト化
    robots = [Robot(id=i, role='explorer',
                    pose=np.array([random.uniform(0, 100),
                                   random.uniform(0, 100),
                                   random.uniform(-math.pi, math.pi)]).T,
                    max_vel=MAX_VEL, field=FIELD)
              for i in range(NUM_BOTS)]
    
    # エージェント（コイツがロボットの動きを決める）のオブジェクト化
    estimators = [Mcl(init_pose=robots[i].pose, num=100)
                  for i in range(NUM_BOTS)]                 # 各エージェントに搭載する自己位置推定器の定義
    agents = [EstimationAgent(id=i, nu=0, omega=0, estimator=estimators[i])
              for i in range (NUM_BOTS)]                    # 各エージェントを定義

    # すべてのロボットにエージェントやセンサを搭載して、環境に登録する
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