# パッケージのインポート
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse

# オリジナルパッケージのインポート
from robot import *
from mcl import *


# 中心位置pに楕円を描く関数
def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)                          # 共分散行列の固有値の算出
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0])/math.pi*180  # 楕円の傾きの算出
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]),
                   angle=ang, fill=False, color="blue", alpha=0.5)  # 楕円のオブジェクトを返す

def matM(nu, omega, time, stds):
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time,
                    stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])

def matA(nu, omega, time, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[(stw-st)/omega, -nu/(omega**2)*(stw-st)+nu/omega*time*ctw],
                     [(-ctw+ct)/omega, -nu/(omega**2)*(-ctw+ct)+nu/omega*time*stw],
                     [0, time]])

def matF(nu, omega, time, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0,2] = nu/omega*(math.cos(theta + omega*time) - math.cos(theta))
    F[1,2] = nu/omega*(math.sin(theta + omega*time) - math.sin(theta))
    return F

# カルマンフィルタのクラス
class KalmanFilter:
    def __init__(self, envmap, init_pose,
                 motion_noise_stds={"nn":0.8,"no":0.001,"on":0.005,"oo":0.2}):
        self.belief = multivariate_normal(mean=np.array([init_pose[0], init_pose[1], init_pose[2]]),
                                          cov=np.diag([1e-10, 1e-10, 1e-10]))
        self.motion_noise_stds = motion_noise_stds
        self.pose = self.belief.mean

    def motion_update(self, nu, omega, time):
        # omegaが小さすぎてゼロに丸められるとエラーが出るため
        if abs(omega) < 1e-5:
            omega = 1e-5

        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, self.belief.mean[2])
        F = matF(nu, omega, time, self.belief.mean[2])
        new_cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)
        new_mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean)
        self.belief = multivariate_normal(mean=new_mean, cov=new_cov)
        self.pose = self.belief.mean
        
    def observation_update(self, id, observation):
        pass

    def draw(self, ax, elems):
        # xy平面上で誤差3シグマ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        # theta方向の誤差3シグマ範囲
        x,y,c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2,2])*3
        xs = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys = [y + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)



# このファイルを直接実行した場合はここからスタートする
if __name__=='__main__':
    
    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600                     # フィールド1辺長さ[m]
    SIM_TIME = 500                  # シミュレーション総時間 [sec]
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
    # m.append_landmark(Landmark(150, 150, 0))
    # m.append_landmark(Landmark(150, 450, 0))
    # m.append_landmark(Landmark(450, 150, 0))
    # m.append_landmark(Landmark(450, 450, 0))
    

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
    estimators = [KalmanFilter(m, init_pose=robots[i].pose)
                  for i in range(NUM_BOTS)]                 # 各エージェントに搭載する自己位置推定器の定義
    agents = [EstimationAgent(time_interval=TIME_STEP,
                              id=robots[i].id, role=robots[i].role, nu=2.0, omega=10.0/180*math.pi, 
                              robot=robots[i], allrobots=robots, estimator=estimators[i])
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