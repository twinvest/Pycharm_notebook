import numpy as np
import pandas as pd

df1 = pd.DataFrame([{'employee': 'Bob', 'Group': 'Accounting'}, {'employee': 'Jake', 'Group': 'Engineering'},
                                   {'employee': 'Lisa', 'Group': 'Engineering'}, {'employee': 'Sue', 'Group': 'HR'}])

df2 = pd.DataFrame([{'employee': 'Bob', 'hire_date': 2008}, {'employee': 'Jake', 'hire_date': 2012},
                                  {'employee': 'Lisa', 'hire_date': 2004}, {'employee': 'Sue', 'hire_date': 2014}])

# df3 =  pd.DataFrame([{'employee': 'Bob', 'hire_date': 2008}, {'employee': 'Jake', 'hire_date': 2012},
#                                   {'employee': 'Lisa', 'hire_date': 2004}, {'employee': 'Sue', 'hire_date': 2014}])
# print(df1)
# print(df2)
# df3 = pd.merge(df1, df2)
# print(df3)
df4 = pd.DataFrame([{'Group': 'Accounting', 'Supervisor': 'Carly'}, {'Group': 'Engineering', 'Supervisor': 'Guido'},
                                   {'Group': 'HR', 'Supervisor': 'Steve'}])
# df3 = df3.drop(['hire_date'], axis=1)
# print(df3)
# print(df4)
# print(pd.merge(df3, df4))

df5 = pd.DataFrame([{'Group': 'Accounting', 'Skills': 'math'}, {'Group': 'Accounting', 'Skills': 'spreadsheets'},
                                   {'Group': 'Engineering', 'Skills': 'coding'}, {'Group': 'Engineering', 'Skills': 'Linux'},
                                   {'Group': 'HR', 'Skills': 'spreadsheets'}, {'Group': 'HR', 'Skills': 'organization'}])

print(pd.merge(df1, df5))


# df6 = pd.DataFrame([{'name': 'Bob', 'Salary': 70000}, {'name': 'Jake', 'Salary': 80000},
#                                   {'name': 'Lisa', 'Salary': 120000}, {'name': 'Sue', 'Salary': 90000}])

df6 = pd.DataFrame([{'name': 'Bob', 'Salary': 70000}, {'name': 'Jake', 'Salary': 80000},
                                  {'name': 'Lisa', 'Salary': 120000}, {'name': 'Sue', 'Salary': 90000}])

tmp = pd.merge(df1, df6, left_on='employee', right_on='name')
print(tmp)
tmp = tmp.drop('name', axis=1)
print(tmp)

# print(pd.merge(df1, df2, left_index=True, right_index=True))
# print(df1)
# print(pd.merge(df1, df6, left_index=True, right_on='name'))


df1a = pd.DataFrame([{'Group': 'Accounting'}, {'Group': 'Engineering'}, {'Group': 'Engineering'}, {'Group': 'HR'}],
                                index=['Bob', 'Jake', 'Lisa', 'Sue'])

# df2a = pd.DataFrame([{'hire_date': 2008}, {'hire_date': 2012}, {'hire_date': 2004}, {'hire_date': 2014}],
#                    index=['Bob', 'Jake', 'Lisa', 'Sue'])
# print(df1a)
# print(df2a)
# df1a.index.name = 'employee'
# print(pd.merge(df1a, df2a, left_index=True, right_index=True))

# print(pd.merge(df1a, df6, left_index=True, right_on ='name'))

# df8 = pd.DataFrame([{'name': 'Bob', 'Rank': 1}, {'name': 'Jake', 'Rank': 2},
#                                   {'name': 'Lisa', 'Rank': 3}, {'name': 'Sue', 'Rank': 4}])
# df9 = pd.DataFrame([{'name': 'Bob', 'Rank': 3}, {'name': 'Jake', 'Rank': 1},
#                                   {'name': 'Lisa', 'Rank': 4}, {'name': 'Sue', 'Rank': 2}])
# #
# print(pd.merge(df8, df9, on='name'))

# df10 = pd.DataFrame([{'name': 'Peter', 'Food': 'Fish'}, {'name': 'Paul', 'Food': 'Beans'},
#                                   {'name': 'Mary', 'Food': 'bread'}])
#
# df11 = pd.DataFrame([{'name': 'Mary', 'Drink': 'Wine'}, {'name': 'Joseph', 'Drink': 'Beer'}])
#
# print(pd.merge(df10, df11, how='outer'))
# print(pd.merge(df10, df11))
# print(pd.merge(df10, df11, how='left'))