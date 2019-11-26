import numpy as np
# list_var = [0,1,2,3,4]
# arr_var = np.array(list_var, dtype = 'int64')
# print(arr_var) #numpy의 array를 출력하면 리스트와 다르게 ','가 없음
# print(arr_var.dtype)

# a=[1,3,5]
# b=[2,4,6]
# result = []
# for ai, bi in zip(a, b):
#     result.append(ai*bi)
# print(result)

# a = np.array([[1, 4, 2, 3, 4, 5],[2, 5, 6,5,2,1]])
# b = np.arange(1, 7, 1)
# c=np.array([3.5,2.4])
# d= np.ones_like(a) * 2.2
# result = 3*a - d
# print(a)
# print(b)
# print(c)
# print(d)
# print(result)
#
#
# A = np.array([range(11, 17), range(17, 23), range(23, 29), range(29, 35)])
# print(A)
# print(A[:2, 3::])
# print(A[:4, 4::4])

# c=np.array([[1,2,3],[4,5,6]])
# print(c%2 == 0)
# print(c[c%2 == 0])
# c[c%2 == 0] = 100
# print(c)
b= np.array([1,3,5,7, 2,4,6,8, 3,6,9,12])
a=np.array([1,3,5])

print(b[a])
