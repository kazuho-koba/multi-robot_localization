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
    def observation_update(self, observation, id, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            is_informed = False     # 今時ステップに基地局から情報を貰えたかどうか

            # ランドマークを検知したら、その観測結果に従って重みを更新する
            if d[0] == 'landmark':
                obs_pos = d[1][0:2]     # 第3要素は対象の向き（ランドマークの場合は使わないデータ）なので除外
                obs_id = d[2]

                # パーティクルの位置と地図からランドマークの距離と方角を算出(自己位置の候補であり確定的な位置座標であるパーティクルの位置から
                # 既知のランドマークを見たらどう見えるか、という演算なので、CameraでなくIdealCameraの観測関数を呼び出している)
                pos_on_map = envmap.landmarks[obs_id].pose
                particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

                # 尤度の計算
                distance_dev = distance_dev_rate * particle_suggest_pos[0]
                cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
                # 尤度に従って重みを更新する。.PDF(arg)はargを得る確率を確率密度関数（PDF、この場合はその前に
                # 記載した多変量正規分布によって得られるPDF）返す。今回はobs_posという「ロボットから見たランドマークの距離、視線角（観測実績値）」が
                # 得られる確率を、PDFが平均値particle_suggest_pos（今考えてるパーティクルから見た場合に得られるはずの距離と視差角）、共分散行列cov（センサ特性で決まる）
                # で表現される多変量正規分布であるとして、得ている。
                self.weight *= multivariate_normal(mean=particle_suggest_pos[0:2], cov=cov).pdf(obs_pos)
            
            
            # 基地局から、基地局による自機位置観測結果を貰ったら、その情報に従って重みを更新する
            if envmap.robots[id].informed_time == envmap.robots[id].current_time:
                is_informed = True  # 情報提供を受けたフラグをオンに 

                # 基地局からもらった情報には自分の向きも含まれる（1つのランドマークを観測するだけでは、
                # ロボット自身の向きの推測は不可能で、自己位置推定結果のパーティクルがランドマーク周りに
                # ドーナツ状に分布することになる）

                # ロボットの位置と姿勢（基地局が観測して通知してきたデータ）
                obs_x, obs_y, obs_theta = envmap.robots[id].informed_pose[0], envmap.robots[id].informed_pose[1], envmap.robots[id].informed_pose[2]

                # 尤度計算に必要な情報を計算する
                distance_dev_x = distance_dev_rate * obs_x
                distance_dev_y = distance_dev_rate * obs_y
                cov = np.diag(np.array([distance_dev_x**2, distance_dev_y**2, direction_dev**2]))

                # パーティクル位置と基地局からの観測データの差を計算する
                diff = np.array([obs_x - self.pose[0], obs_y - self.pose[1], obs_theta - self.pose[2]])
                
                # それに基づいて重みを更新する
                # ランドマークを観測した場合の応用で、diff（基地局が観測したロボットの位置・姿勢と、今検討対象としているパーティクルの位置・姿勢の差分）
                # が得られる確率を、PDFが平均ゼロ、共分散がセンサ特性で決まる多変量正規分布であるとして計算している。
                self.weight *= multivariate_normal(mean=[0, 0, 0], cov=cov).pdf(diff)


                ######## 重み更新の別解（やることは上と同じ） ##################
                # obs_x, obs_y, obs_theta（基地局が観測したロボットの位置・姿勢）が得られる確率を、PDFが
                # 平均がパーティクルの位置・姿勢、共分散がセンサ特性によって決まるものとして計算し重みに乗じる
                # self.weight *= multivariate_normal(mean=np.array([self.pose[0], self.pose[1], self.pose[2]]),
                #                                    cov = cov).pdf(np.array([obs_x, obs_y, obs_theta]))
                ########################################
                

    # 自己位置推定器から呼び出され、相互観測の結果として自身のパーティクルの尤度を計算する
    def mutual_observation_update(self, pose_obs, confidence_diff, lack_of_confidence_others,
                                  bs_comm_time_other, current_time,
                                  distance_dev_rate, direction_dev):
        
        # ロボットの位置と姿勢（相手ロボが観測して通知してきたデータ
        obs_x, obs_y, obs_theta = pose_obs[0], pose_obs[1], pose_obs[2]

        # 尤度計算に必要な情報を計算する（相手からの観測の誤差を考慮
        distance_dev_x = distance_dev_rate * obs_x
        distance_dev_y = distance_dev_rate * obs_y
        cov = np.diag(np.array([distance_dev_x**2, distance_dev_y**2, direction_dev**2]))
        
        # パーティクル位置と相手ロボットの観測ベースの自己位置の差を計算する
        diff = np.array([obs_x - self.pose[0], obs_y - self.pose[1], obs_theta - self.pose[2]])
        
        # 彼我の自己位置推定結果に対する確信度の差から重み付けをスケーリングする
        diff_scale = 1.0                                    # 全体の係数
        lack_of_confidence_others += 1.0                    # 相手の確信度が高い（値が小さい）とここがゼロになってしまい後でスケールが無限になってしまう
        adjust_scale_conf = 1/(1+(75*lack_of_confidence_others)) # 相手の自己位置推定結果に対する自身のなさに応じた係数（元の値は高いほど自信がないので逆数を取る）
        adjust_scale_bs = np.exp(-(current_time-bs_comm_time_other)/75)
        scaling_factor = np.exp((confidence_diff/diff_scale) * adjust_scale_conf * adjust_scale_bs)
        # print(confidence_diff, adjust_scale, scaling_factor)

        # パーティクルの重みを変更する倍数を計算
        weight_update = multivariate_normal(mean=[0,0,0], cov=cov).pdf(diff)
        # スケーリングファクタと掛け合わせて重みを更新
        self.weight *= weight_update ** scaling_factor
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

    # エージェントから呼び出されてセンサ情報（ランドマークの観測＆基地局からの観測結果通知）に基づきパーティクル情報を更新する
    def observation_update(self, id, observation):
        for p in self.particles:
            p.observation_update(observation, id, self.map, self.distance_dev_rate, self.direction_dev)
        
        # 以下の処理はmutual_observation_updateで実行
        # # この時点で尤度最大のパーティクル情報を取得
        # self.set_ml()

        # # パーティクルのリサンプリングを実施する（何も観測していない時に実行しても
        # # その時点の自分の認識のバイアスを肥大化させるだけなので、観測物が存在するときのみ実行する
        # # if len(observation) > -1:
        # #     self.resampling()   

        # # 修正したリサンプリングでは上記の心配がない
        # self.resampling_mod()

        pass

    # エージェントから呼び出されてセンサ情報（ロボット同士の相互計測）に基づきパーティクルの重みを更新する
    def mutual_observation_update(self, id, is_locked,
                                  lack_of_confidence_self, lack_of_confidence_others,
                                  bs_comm_time, current_time):

        # 誰の情報に基づいて自分がパーティクルの重みを更新したかを控えておくリスト
        updated = {key:False for key in range(len(self.map.robots))}

        # 以下、相手の情報は相手が計算して結果をこっちに通信で教えてくるという想定
        for i in range(len(self.map.robots)):
            otherbot = self.map.robots[i]   # 他のロボットの情報            
            obs_others = otherbot.obs       # 他のロボットの観測結果
            
            if obs_others != None:
                for d in obs_others:
                    # 他のロボットが自身を観測した結果を通知してきて、かつそのロボットの情報に基づく更新が許可されている場合
                    if d[0] == 'robot' and d[2] == id and is_locked[otherbot.id] == False \
                        and lack_of_confidence_others[otherbot.id] != None:
                        # そのロボットの自己位置推定の確信度（の低さ）
                        otherbot_lack_of_conf = lack_of_confidence_others[otherbot.id]  
                        
                        # 相手の観測ベースでの自己位置を取得（誤差が載った相手の自己位置推定結果＋誤差が載った自機の観測結果）
                        pose_obs = np.array([otherbot.agent.estimator.pose[0] + d[1][0]*math.cos(d[1][1]),
                                             otherbot.agent.estimator.pose[1] + d[1][0]*math.sin(d[1][1]),
                                             d[1][2]])                      

                        # 彼我の確信度の差に応じてパーティクル情報を更新
                        confidence_diff = lack_of_confidence_self - otherbot_lack_of_conf   # 確信度の差（元データは確信度の低さなので、これは値が大きいほど、相手の方が自信がある）
                        if confidence_diff > 0 and bs_comm_time[id] < bs_comm_time[otherbot.id]:
                            # 相手の方が自己位置推定に自信がある時、相手の情報をベースに自身の自己位置推定を更新する
                            for p in self.particles:
                                p.mutual_observation_update(pose_obs, confidence_diff, otherbot_lack_of_conf,
                                                            bs_comm_time[otherbot.id], current_time,
                                                            self.distance_dev_rate, self.direction_dev)
                            pass

                            # 相手の情報に基づいて情報を更新したことをメモ
                            updated[otherbot.id] = True

        # この時点で尤度最大のパーティクル情報を取得
        self.set_ml()

        # パーティクルのリサンプリングを実施する（何も観測していない時に実行しても
        # その時点の自分の認識のバイアスを肥大化させるだけなので、観測物が存在するときのみ実行する
        # if len(observation) > -1:
        #     self.resampling()   

        # 修正したリサンプリングでは上記の心配がない
        self.resampling_mod()

        return updated
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
    def __init__(self, time_interval, id, role, nu, omega,
                 robot=None, allrobots=None, estimator=None):
        # 継承元のAgentクラスの初期化処理を実行
        super().__init__(id, role, nu, omega, robot, allrobots)
        self.estimator = estimator
        self.time_interval = time_interval

        # ロボットに1ステップ前に指示した移動量（これにノイズを載せてパーティクルの移動量にする）
        self.prev_nu = 0.0
        self.prev_omega = 0.0
        
        # 相互計測による自己位置推定更新をロックするフラグ
        self.lock_mutual_obs_localization = {key:False for key in range(len(self.allrobots))}
        self.obs_robot_time = {key:False for key in range(len(self.allrobots))} # 最後に相手を観測した時刻
        self.bs_comm_time = {key:0 for key in range(len(self.allrobots))}       # 基地局から最後に情報を受けた時刻

    # エージェントが意思決定をするメソッド（継承元のメソッドを上書き
    def decision(self, observation=None):
        # パーティクルの動きを演算する
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        
        # 今時ステップにロボットに指示した移動量（コレ自体は継承元クラスで目標位置をベースに演算）を控えておく
        self.prev_nu, self.prev_omega = self.nu, self.omega

        # センサ情報を取得して自己位置推定を更新（ランドマークの観測と基地局からの観測結果の通知を受けた更新）
        self.estimator.observation_update(self.id, observation)
        
        if hasattr(self.estimator, 'particles') == True:
            # センサ情報を取得して自己位置推定を更新（ロボット同士の相互計測
            # まずそれぞれの自己位置推定結果の確信度の低さ（パーティクルの分布の広さ）を計算する（計測範囲外＝通信範囲外のものについては空になる）
            lack_of_confidence_self = self.compute_confidence(self.estimator.particles)
            # print(self.id, lack_of_confidence_self)
            lack_of_confidence_others = {key:1e5 for key in range(len(self.allrobots))}
            for d in self.robot.obs:
                if d[0] == 'robot':
                    # 観測できたものは通信ができるものとして、相手が計算したものをこっちに送ってくれたという想定
                    lack_of_confidence_others[d[2]] = self.compute_confidence(self.allrobots[d[2]].agent.estimator.particles)
                    self.bs_comm_time[d[2]] = self.allrobots[d[2]].informed_time
            self.bs_comm_time[self.id] = self.robot.informed_time

            # 計算した確信度の低さをベースに、パーティクルの重み更新（確信度が高いロボットに基づき低い方を更新する）
            # 返り値はID別の辞書型リスト（Trueなら、その相手の情報に基づきパーティクルの重みを更新した）
            # updated = {key:False for key in range(len(self.allrobots))}
            updated = self.estimator.mutual_observation_update(self.id, self.lock_mutual_obs_localization,
                                                            lack_of_confidence_self, lack_of_confidence_others,
                                                            self.bs_comm_time, self.robot.current_time)
            # ある相手の情報に基づいて自身のパーティクル重みを更新していたら、それ以降はその相手の情報に基づいた
            # 更新もしないし、相手も自身の情報に基づく更新は行わない
            for i in range(len(self.allrobots)):
                if updated[i] == True:
                    self.lock_mutual_obs_localization[i] = True
                    self.allrobots[i].agent.lock_mutual_obs_localization[self.id] = True

            # 前述のロックは、相手が観測範囲から外れたら解除する
            for i in range(len(self.allrobots)):
                for d in self.robot.obs:
                    if d[0] == 'robot' and d[2] == i:
                        self.obs_robot_time[i] = self.robot.current_time
                # 2秒以上観測がなければロック解除
                if self.obs_robot_time[i] < self.robot.current_time - 1:
                    self.lock_mutual_obs_localization[i] = False

        # 返り値
        return self.nu, self.omega
    
    # パーティクルの分布の広さを計算するメソッド
    def compute_confidence(self, particles):
        # パーティクルの各要素を取得
        x = np.array([p.pose[0] for p in particles])
        y = np.array([p.pose[1] for p in particles])
        theta = np.array([p.pose[2] for p in particles])

        # それぞれの分散を計算
        var_x = np.var(x)
        var_y = np.var(y)
        var_theta = np.var(theta)

        # 分散を正規化
        var_x /= self.robot.field**2    # xの範囲は0から600
        var_y /= self.robot.field**2    # yの範囲はーー
        var_theta /= (2*np.pi)**2       # thetaの範囲は-piからpi

        # 正規化標準偏差の和を自信のなさの指標とする
        lack_of_confidence = np.sqrt(var_x) + np.sqrt(var_y) + np.sqrt(var_theta)

        return lack_of_confidence



    # 自身が基地局エージェントの場合に実行する処理
    def bs_task(self):
        # 他のロボットの観測結果を抽出する
        observed_robot = [o for o in self.robot.obs if o[0]=='robot']   # 検知したロボットのリストを観測リストから取得
        
        # 各ロボットの位置推定結果を計算し、そのロボットに教えてあげる
        for obs_bot in observed_robot:
            est_pose = np.array([obs_bot[1][0]*math.cos(obs_bot[1][1]),
                                 obs_bot[1][0]*math.sin(obs_bot[1][1]),
                                 obs_bot[1][2]]).T
            self.allrobots[obs_bot[2]].informed_pose = est_pose
            self.allrobots[obs_bot[2]].informed_time = self.robot.current_time
            
    
    # 描画に関する処理（エージェントが想定する自己位置に関する信念を描写する）
    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

        # 尤度最大のパーティクルの姿勢を描画
        x,y,t = self.estimator.pose     # 座標の取得
        s = "({:.1f}, {:.1f}, {})".format(x, y, int(t*180/math.pi)%360)
        # elems.append(ax.text(x, y+0.1, s, fontsize=10))
        elems.append(ax.quiver(x, y, math.cos(t), math.sin(t),
                               color='red', scale=15, alpha=0.5))


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
    # m.append_landmark(Landmark(80, 0, 0))
    # m.append_landmark(Landmark(0, 80, 0))
    

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
    estimators = [Mcl(m, init_pose=robots[i].pose, num=100,
                      motion_noise_stds={"nn":0.8,"no":0.001,"on":0.005,"oo":0.12})
                  for i in range(NUM_BOTS)]                 # 各エージェントに搭載する自己位置推定器の定義
    agents = [EstimationAgent(time_interval=TIME_STEP,
                              id=robots[i].id, role=robots[i].role, nu=0, omega=0, 
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