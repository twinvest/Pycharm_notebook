import pandas as pd
from selenium import webdriver

df = pd.read_csv('adult.dat'
                 'a', header=None) #csv파일을 읽어와라 헤더는 논.
#data basic
print(df.size)
print(df.shape)
print(df.columns)

df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'wage']
print(df.columns)
print(df.dtypes)
print(df.head()) #데이터의 상위 다섯개
print(df.tail()) #데이터의 하위 다섯개

# data summary
print(df.describe()) #요약한 정보를 제공해주는 함수
print(df.mean())     #numerical type에 대해서만 평균값을 제공해 6개가 나온거임
print(df.mode())

# Details
print(df.education.unique())
print(df.education.value_counts())
print(df['wage'].value_counts())
print(df.groupby(['wage'])['age'].mean())
print(df.groupby(['wage'])['age'].std())
print(df['capital-gain'].corr(df['age']))