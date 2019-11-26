import numpy as np
from matplotlib import pyplot as plt
# x = np.linspace(-np.pi, np.pi, 1000)
#
# mean = 0 # 평균
# std = 1 # 표준편차
# data = np.random.normal(mean, std, 10000)
# plt.hist(data, bins=100) # 나누는 구간 개수 (100개 정도로 더 잘게 나누어 보라는 의미)
# plt.show()

rand_array = np.random.randn(2, 1000)
uni_array = np.random.rand(2, 1000) * 6 