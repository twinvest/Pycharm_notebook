import numpy as np
import pandas as pd

# df3 = pd.DataFrame([{'A': 'A0', 'B': 'B0'}, {'A': 'A1', 'B': 'B1'}])
# df4 = pd.DataFrame([{'C': 'C0', 'D': 'D0'}, {'C': 'C1', 'D': 'D1'}])
#
# print(pd.concat([df3, df4]))
# print(pd.concat([df3, df4], axis=1))
# print(pd.concat([df3, df4], ignore_index=True))

df7 = pd.DataFrame([{'A': 'A1', 'B': 'B1', 'C': 'C1'}, {'A': 'A2', 'B': 'B2', 'C': 'C2'}], index=[1, 2])
df8 = pd.DataFrame([{'B': 'B3', 'C': 'C3', 'D': 'D3'}, {'B': 'B4', 'C': 'C4', 'D': 'D4'}], index=[3, 4])
#print(pd.concat([df7, df8]))
# print(pd.concat([df7, df8], join='inner'))
# print(pd.concat([df7, df8], join_axes=[df7.columns]))
print(pd.concat([df7, df8], axis=1 , join_axes=[df7.index]))
