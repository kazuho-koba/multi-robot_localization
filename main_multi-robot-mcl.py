from mcl import *

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