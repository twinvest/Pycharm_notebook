import pandas as pd
import numpy as np

s1 = pd.Series([17, 17])
s2 = pd.Series([3, 11])
# s1 = pd.Series({"0" : 17, "1" : 17})
# s2 = pd.Series({"0" : 3, "1" : 11})

df = pd.DataFrame({'A' : s1, 'B' : s2})
print(df)
print(df-2) #이렇게 하면 dt안에 있는 모든 요소에 -2가 된다.
print(df - np.array([2,3]))  #각 로우에 [2, 3]을 빼준다.

print('=======================================')
print(df.iloc[0]) #[17, 3]
print(df.iloc[1]) #[17, 11]
print('=======================================')
print(df - df.iloc[0])
print('=======================================')

df2 = pd.DataFrame([{'B':0, 'A':1, 'C':2}, {'B':5, 'A':8, 'C':7}, {'B':3, 'A':0, 'C':3}])
print(df2)
print('=======================================')
print(df + df2)
print('=======================================')
print(df/df2)