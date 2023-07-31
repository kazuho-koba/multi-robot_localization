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

def matH(pose, landmark_pos):
    mx, my = landmark_pos[0:2]          # ランドマークの位置
    mux, muy, mut = pose                # （多分）自分の位置・姿勢
    q = (mux - mx)**2 + (muy - my)**2

    return np.array([[mux-mx/np.sqrt(q), (muy-my)/np.sqrt(q), 0.0],
                     [(my - muy)/q, (mux - mx)/q, -1.0]])

def matQ(distance_dev, direction_dev):
    # 元コードと異なり、観測対象の向きまで観測できるようにしたため、3次元に拡張している。第1が相手までの距離、第2が相手への視差角、
    # 第3が相手の向きの検知に関する誤差を示しているが、今は第2と同じ値を流用して仮実装した。
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


# カルマンフィルタのクラス
class KalmanFilter:
    def __init__(self, envmap, init_pose,
                 motion_noise_stds={"nn":0.8,"no":0.001,"on":0.005,"oo":0.2},
                 distance_dev_rate = 0.14, direction_dev = 0.05):
        self.belief = multivariate_normal(mean=np.array([init_pose[0], init_pose[1], init_pose[2]]),
                                          cov=np.diag([1e-10, 1e-10, 1e-10]))
        self.motion_noise_stds = motion_noise_stds
        self.pose = self.belief.mean
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

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
        
    # 観測に基づく信念の更新
    def observation_update(self, id, observation):
        
        for d in observation:
            # 観測したものがランドマークだった場合
            if d[0] == 'landmark':
                z = d[1]            # ランドマークの位置
                obs_id = d[2]       # ランドマークのID

                H = matH(self.belief.mean, self.map.landmarks[obs_id].pose)
                estimated_z = IdealCamera.observation_function(self.belief.mean, 
                                                               self.map.landmarks[obs_id].pose) # 自己位置（推定値の中心）からランドマーク（真値）を見たらどう見えるハズか？
                Q = matQ(estimated_z[0]*self.distance_dev_rate, self.direction_dev)
                K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T)))
                
                # 信念分布の更新（元のコードと異なり観測結果が距離、視線角、相手の姿勢の3項目なので注意）
                new_mean = self.belief.mean + K.dot(z[0:2] - estimated_z[0:2])
                new_cov = (np.eye(3) - K.dot(H)).dot(self.belief.cov)
                # 多変数共分散のmean,covを直接書き換えることはできないので、新しい平均と共分散で分布を作り直す
                self.belief = multivariate_normal(mean=new_mean, cov=new_cov)
                self.pose = self.belief.mean            

            # 基地局から、基地局による自己位置観測結果をもらったら、その情報に従って信念分布を更新する
            if self.map.robots[id].informed_time == self.map.robots[id].current_time:
                # ロボットの位置と姿勢（基地局が観測・通知してきたデータ
                obs_x, obs_y, obs_theta = self.map.robots[id].informed_pose[0], self.map.robots[id].informed_pose[1], self.map.robots[id].informed_pose[2]


    def draw(self, ax, elems):
        # xy平面上で誤差3シグマ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 2)
        elems.append(ax.add_patch(e))

        # theta方向の誤差3シグマ範囲
        x,y,c = self.belief.mean
        sigma2 = math.sqrt(self.belief.cov[2,2])*2
        xs = [x + 50*math.cos(c-sigma2), x, x + 50*math.cos(c+sigma2)]  # 3点のx座標
        ys = [y + 50*math.sin(c-sigma2), y, y + 50*math.sin(c+sigma2)]  # 3点のy座標
        elems += ax.plot(xs, ys, color='blue', alpha=0.5)               # 上記3点を順に結ぶ折れ線



# このファイルを直接実行した場合はここからスタートする
if __name__=='__main__':
    
    ################################
    # シミュレーションの設定
    NUM_BOTS = 4                    # ロボット総数
    MAX_VEL = np.array([2.0, 1.0])  # ロボット最大速度（[m/s], [rad/s]）
    FIELD = 600                     # フィールド1辺長さ[m]
    SIM_TIME = 500                  # シミュレーション総時間 [sec]
    TIME_STEP = 1                   # 1ステップあたり経過する秒数
    SAVE_VIDEO = True              # 動画ファイルを保存
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
    # m.append_landmark(Landmark(75, 75, 0))
    # m.append_landmark(Landmark(150, 75, 0))
    # m.append_landmark(Landmark(75, 150, 0))
    # m.append_landmark(Landmark(150, 150, 0))
    m.append_landmark(Landmark(300, 300, 0))
    m.append_landmark(Landmark(150, 450, 0))
    m.append_landmark(Landmark(450, 150, 0))
    m.append_landmark(Landmark(450, 450, 0))
    

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