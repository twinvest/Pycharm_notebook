# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("basic_Info.csv", encoding='UTF-8')
#
# id = df['직원id']
# hire_form = df['고용형태'] #상용직 일용직
# position = df['직종'] #계약 정규 촉탁 파견 null값
#
# # 정규직=상용직
# # 비정규직=계약직=일용직
#
# path='./new_staff_division.csv'
# # department = df1['부서'].isnull().sum()
#
# table=pd.DataFrame({'id': id, 'hire_form':hire_form, 'position': position})
# table['category'] = None
# # print(table)
#
# table.loc[((table['position'] == None) & (table['hire_form'] == '상용직')) | table, 'category'] = '정규직'
# table.loc[(table['position'] == None) & (table['hire_form'] == '일용직'), 'category'] = '비정규직'
# print(table)
#
#
# # table = pd.DataFrame({'id':id, 'position':position_number})
# # table.to_csv(path, encoding="ms949", mode="w", index=False)

import pandas as pd
import numpy as np

df = pd.read_csv("./employee.csv", encoding='euc-kr')

id = df['직원id']
hire_form = df['고용형태']
position = df['직종']

path='./new_staff_division.csv'

table=pd.DataFrame({'id':id, 'position':position, 'hire_form':hire_form})
table['result'] = None

table.loc[(table.position == '정규직'), 'result'] = '정규직'
table.loc[(table.position == None) & (table.hire_form == '상용직'), 'result'] = '정규직'
table.loc[(table.hire_form == '일용직'), 'result'] = '비정규직'
table.loc[(table.position == '촉탁직'), 'result'] = '촉탁직'

print(table)
df1 = pd.DataFrame({'id':id, 'result':table.result})
df1.to_csv(path, encoding="ms949", mode="w", index=False)