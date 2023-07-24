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

# クラスとして自分で定義した諸々の読み込み
from ideal_robot import *

# より現実的なロボットのクラスをIdealRobotを継承して実装する
class Robot(IdealRobot):
    pass