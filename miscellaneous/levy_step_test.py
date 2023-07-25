# import numpy as np
# import matplotlib.pyplot as plt

# # パラメータの設定
# alpha = 2.5         # パワーローの指数
# min_step = 30       # ステップサイズの最小値
# max_step = 500      # ステップサイズの最大値
# num_samples = 10000 # 生成するサンプルの数

# # 乱数の生成
# step_sizes = (max_step - min_step) * (1 - np.random.power(alpha, num_samples)) + min_step

# # ヒストグラムの表示
# plt.hist(step_sizes, bins=50, alpha=0.5, color='g')
# plt.xlabel('Step size')
# plt.ylabel('Frequency')
# plt.title('Histogram of step sizes drawn from a power-law distribution')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 定数の設定
alpha = 1.2
d_min = 30
d_max = 500
num_samples = 10000

# 一様分布からランダムにサンプルを生成
uniform_samples = np.random.uniform(0, 1, num_samples)

# 逆変換サンプリングを使用して目的の分布に従ったサンプルを生成
power_law_samples = d_min * ((1 - uniform_samples) ** (1 / (1 - alpha)))

# ステップサイズの最大値に制限を加える
power_law_samples = power_law_samples[power_law_samples <= d_max]

# ヒストグラムをプロット
plt.hist(power_law_samples, bins=100)
plt.show()


'''
Chat GPTに聞いた
Kazuho
たとえば、ステップサイズdに対して、p(d) = gannma * d^(- alpha)という確率分布にしたがってdをドローするコードはどう書けばいいですか？dの範囲は前の質問と同じです。

ChatGPT
この問題は一見すると直感的ではないかもしれませんが、p(d) = gamma * d^(-alpha)という確率分布に従うステップサイズdを生成する方法はあります。ここで、gammaは正規化定数で、alphaはパワーローの指数です。

この分布はパワーロー分布の一種で、alphaが大きいほど分布は左に偏り（つまり小さいdの頻度が高くなる）、alphaが小さいほど分布は右に偏ります（つまり大きなdの頻度が高くなる）。

この分布から直接サンプルを生成するには、逆変換サンプリングという手法を用います。まず、累積分布関数（CDF）を求め、その逆関数（quantile function）を求めます。そして、[0,1]の一様分布からランダムに数値を生成し、それを逆関数に入れることで目的の分布に従ったサンプルを生成します。

ここではp(d) = gamma * d^(-alpha)のCDFとその逆関数を求めます。この分布のCDFはF(d) = 1 - (d / d_min)^(1 - alpha)であり、逆関数はF_inv(x) = d_min * ((1 - x) ^ (1 / (1 - alpha)))です。

これらの結果を用いてPythonのコードを書くと以下のようになります。

python
Copy code
import numpy as np
import matplotlib.pyplot as plt

# 定数の設定
alpha = 1.2
d_min = 30
d_max = 500
num_samples = 1000

# 一様分布からランダムにサンプルを生成
uniform_samples = np.random.uniform(0, 1, num_samples)

# 逆変換サンプリングを使用して目的の分布に従ったサンプルを生成
power_law_samples = d_min * ((1 - uniform_samples) ** (1 / (1 - alpha)))

# ステップサイズの最大値に制限を加える
power_law_samples = power_law_samples[power_law_samples <= d_max]

# ヒストグラムをプロット
plt.hist(power_law_samples, bins=100)
plt.show()
このコードでは逆変換サンプリングを使用してp(d) = gamma * d^(-alpha)という分布に従うステップサイズを生成しています。最後にpower_law_samplesをd_max以下の値に制限しています。このようにすることでdの範囲が[d_min, d_max]になります。

このコードはalphaが1より大きい場合にのみ有効です。alphaが1以下の場合、p(d) = gamma * d^(-alpha)は正規化できないため、この方法は使用できません。その場合、他の手法（例えば、リジェクションサンプリング）を使用する必要があります。
'''