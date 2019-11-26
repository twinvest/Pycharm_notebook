import numpy as np
from matplotlib import pyplot as plt
"""
안타, 홈런, 볼넷, 삼진, 도루, 타율
구분자는 comma를 사용함
data = arr = np.loadtxt('baseball_data.txt', delimiter=',')를 이용

다음의 계산을 수행하고 plotting 할 것
음수값을 0으로 대체
"""

arr = np.loadtxt('text.txt', delimiter=',')
print(arr)
print('==========================================')
arr[arr<0] = 0 #음수값은 0으로 대체
print(arr)
print('==========================================')
arr = arr[arr[:,1]>0] #홈런이 1개 이상인 놈들
print(arr)
print('==========================================')
h, hr, bb, k, sb, avf = arr.T
print(arr)
print('==========================================')
hr_per_h = hr/h
bb_per_h = bb/h
k_per_h = k/h


hr_per_h /= np.max(hr_per_h) #정규화
bb_per_h /= np.max(bb_per_h) #정규화
k_per_h /= np.max(k_per_h)   #정규화

plt.plot(hr_per_h, k_per_h, 'rx')
plt.show()