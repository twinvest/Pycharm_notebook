import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
데이터를 처리하고, 몇가지 정보를 확인하고자 할 때 필요한 몇가지 방법들을 소개하고자 함.
원하는 데이터를 추출하고자할 때
특정 변수의 조건에 대해 새로운 변수를 만들고자할 때
데이터의 전체적인 통계량 및 aggregation을 하고자 할 때
차트를 이용해서 데이터를 확인하고자 할 때
우선, 데이터 중 하나를 입력하여 이에 대한 형태를 확인한다.
"""

data = pd.read_csv('titanic.csv')
print(data.shape)
print(data.head())
"""
1. 원하는 데이터를 추출할 때
특정 조건에 해당하는 데이터 셋을 찾고자 할 때 사용할 수 있음.
"""
#data.loc은 조건을 줄 수 있음. 그리고 해당 조건에 해당하는 데이터를 가지고온다.
#head() 함수는 5개 정도만 확인할 수 있음.
#P클래스는 객실 등급. 1이 제일 좋을걸? SibSp는 Name이라는 사람이랑 같이 동승한 형제/자매 수. Parch는 Name이라는 사람이랑 같이 동승한 부모의 수.
#Fare는 지불한 비용. Cabin은 객실,
survived_data = data.loc[data['Survived']==1]
survived_data.head()
print(survived_data.shape)

#생존하지 못한(class =0)인 데이터셋을 추출하세요
death_data = data.loc[data['Survived']==0]
death_data.head()
print(death_data.shape)


"""
3. 데이터 그룹화 및 통계량 산출
데이터를 특정 변수에 대해서 그룹화를 진행하여 통계량을 비교하고 싶을 수 있다. 이때 사용하는 방법은 groupby 를 이용한다.
"""
print(data.groupby('Survived').count()) #group by에 변수명을 넣어준다. count함수 말고도 mean도 사용할 수 있으며 min max std 등등이 있다.
print(data.groupby(['Survived', 'Pclass'])['Fare'].count()) #그룹을 하나말고 여러개를 사용하고 싶다면?? 리스트의 형태로 컬럼명을 넣어준다.

#연령대별, 성별로 그룹화하여 생존된 숫자를 확인하세요.
data['age_group'] = np.where(data['Age'] < 10, 'young', np.where(data['Age']<20, '10s', np.where(data['Age']<30, '20s', 'else')))
print(data['age_group'])
tw = data.groupby(['age_group', 'Sex'])['Survived'].count()
print(tw)

"""
4. 차트로 확인하기
위에서 진행한 그룹화 과정을 차트로 직접 확인할 수 있다. 
방법은 크게 두가지이며, 하나는 위에서 구한 groupby 데이터 객체를 이용하는 방식과, 기존의 데이터 프레임에서 pivot table을 이용하는 방식이 있다.
일반적으로 matplotlib을 사용한다.
그러나 matplotlib은 처음에는 사용하기 쉽지 않다.
일단, pandas에서 제공하는 plot을 사용해 공부해보자.
"""
#grouped_data=data.groupby(['Survived', 'Pclass'])['Fare'].mean() #그룹바이한 것들(Survived, Places)에 대한 이용요금에 대한 평균치를 보여준다.
#grouped_data.plot(kind='barh')

plt.interactive(True) #stack overflow에서 pycharm에서 plot를 출력하기 위해선 이걸 넣어주라고 함
a = tw.plot(kind='barh') #plot으로는 출력안되넹?
a.show()

# pclass_pivot = data.pivot_table(index=["Pclass"],values="Survived", aggfunc={'Survived' : np.sum}) #default는 평균값이 나온다. 파라미터에 aggfunc={'Survived':np.num}를 추가해보자.
# pclass_pivot.plot.bar()