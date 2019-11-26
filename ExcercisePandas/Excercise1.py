import pandas as pd
import numpy as np
"""
Pandas는 Numpy의 기능을 확장한 패키지이다.
Numpy에 있던 기존 기능을 승계하였다고 볼 수 있다. 1-d array, 2-d array를 각각 
Series와 DataFrame이라는 클래스를 사용하여 기능을 확장하였다.
"""
data = pd.Series([0.25, 0.5, 0.75,1.0]) #list(혹은 array)객체로부터 Series객체를 생성
print(data)

print('=======================================')

population_dict = {'California' : 38332521, "Texas" : 26448193, "NewYork" : 19651127, "Florida" : 19552860, 'Illinois' : 12882135} #Dictionary객체로부터 Series객체를 생성
population = pd.Series(population_dict)
print(population)

print('=======================================')

print("data의 벨류값 : %s" % data.values)
print("population의 벨류값 : %s" % population.values)

print('=======================================')

print(data.index)
print(population.index)

print('=======================================')

"""
1d-array와의 가장 큰 차이점은 index를 명시적으로 정의할 수 있다는 것
지금까지 알고 있던 sequence 객체 및 np array의 경우 모두 가장 처음에 등장하는 아이템의 index 번호가 0이고, 그 이후 1씩 증가하는
암시적인(implicit)인덱스를 가지고 있었다.
Series의 경우 암시적인 인덱스와 더불어 명시적인(explicit)인덱스를 정의할 수 있다. 만약, 객체 생성시 별도의 index 정의가 없다면 암시적인 인덱스를 그대로 사용한다.
int, float, string등을 사용가능하다. 연속될 값일 필요도 없으며, 심지어 중복 또한 허용한다.
암시적 인덱스를 사용했을 때와 같은 문법으로 indexing 및 slicing이 가능하다
"""

print('=======================================')

data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
print(data)

print('=======================================')

print(data[['b','c']])

print('=======================================')

print(data['b':'d'])

print('=======================================')

#dictionary를 이용하면, data와 index 부분에 모두 index가 존재하는 부분에 대해서만 Series가 생성된다.
data = pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
print(data)

print('=======================================')

"""
DataFrame은 numpy의 2-d array와 대응된다(행과 열에 이름이 부여된 2-d array)
Series처럼 명시적 index를 가지고 있다.
또한 각 열이 이름을 갖고 있어, 사용자의 가독성이 array에 비해서 더 높다.
여러 개의 Series가 붙어있는 형태로 생각할 수도 있다.
두 개의 Series로 DataFrame을 생성 시 index 기준으로 병합된다.
"""

print('=======================================')
population_dict = {'California' : 38332521, "Texas" : 26448193, "NewYork" : 19651127, "Florida" : 19552860, 'Illinois' : 12882135} #위에서 이렇게 생성해봤다.
population = pd.Series(population_dict)
area_dict = {'California' : 423967, "Texas" : 695662, "NewYork" : 141297, "Florida" : 170312, 'Illinois' : 149995}
area = pd.Series(area_dict)

states = pd.DataFrame({'1 : population' : population, '2 : area' : area}) #위에서 만든 두 개의 Series들의 결합으로 DataFrame생성
print(states)

print('=======================================')

"""
DataFrame의 주요 Attributes
Values : 실제 데이터를 담고 있는 컨테이너, Numpy array
Index : 실제 데이터의 위치 정보를 담고 있는 object
Columns : index와 유사하나 행이 아닌 열에 대한 정보를 담고 있음
"""

print('=======================================')

dataframe = pd.DataFrame(population, columns=['tw']) #이렇게도 생성가능
print(dataframe)

dataframe2 = pd.DataFrame(np.random.randn(3, 2), columns=['foo', 'bar'], index=['a','b','c']) #np.random.randn()은 가우시안 표준 정규 분포에서 난수 matrix array생성
print(dataframe2)

dataframe3 = pd.DataFrame([{'a':1, 'b':2},{'b':3, 'c':4}]) #dictionary의 key는 column name으로 사용되며 결측치가 있을 경우 NaN가 할당된다
print(dataframe3)

print('=======================================')

"""
기본적으로 Numpy array에서 사용하는 indexing, slicing, masking등 그대로 사용 가능하다.
Series와 DataFrame에서의 indexing/slicing 문법은 같다
먼저, Column-wise selection을 살펴보자. 참고로, Selection이나 연산의 결과물로 얻은 Series를 새로운 Column을 생성한 뒤 할당할 수 있다.
"""

population_dict = {'California' : 38332521, "Texas" : 26448193, "NewYork" : 19651127, "Florida" : 19552860, 'Illinois' : 12882135}
population = pd.Series(population_dict)
area_dict = {'California' : 423967, "Texas" : 695662, "NewYork" : 141297, "Florida" : 170312, 'Illinois' : 149995}
area = pd.Series(area_dict)

states = pd.DataFrame({'population' : population, 'area' : area}) #위에서 만든 두 개의 Series들의 결합으로 DataFrame생성
print(states['area']) #둘의 출력 결과는 동일하다.

print('=======================================')

print(states.area) #둘의 출력 결과는 동일하다.

print('=======================================')

print(states) #추가전
states['density'] = states['population'] / states['area']
print(states) #추가후

print('=======================================')

"""
Row-wise selection을 살펴보자.
DataFrame에서는 data selection을 위해 iloc, loc, ix의 세 가지 함수를 지원한다.
세 가지 방법 모두 2-d array와 비슷한 방식으로 indexing한다.

iloc
: 암시적 인덱스이다. 즉, 0부터 시작하는 인덱싱을 사용하여 data를 선택한다.
  첫 row의 index 번호가 0이며 1씩 증가한다.
  첫 column의 index번호도 마찬가지로 0이며 1씩 증가한다.
  
loc
: 명시적 인덱스를 사용한다. 지금까지 배운 모든 Python indexing 방식과 달리 범위가 폐구간이다.(양쪽모두 inclusive)

ix
: iloc와 loc를 혼합하여 사용할 수 있다.
  그러나, Deprecated 되었으므로 사용하지 않는 것을 권장한다.
"""
print(states.iloc[:3, :2]) #앞이 row이고 뒤가 col이다.
print(states.loc[:'NewYork', :'density']) #마찬가지로 앞이 row이고 뒤가 col이다.
#print(states.ix[:3, 'area']) #이또한 마찬가지

print('=======================================')

"""
loc를 통한 데이터 수정을 살펴보자.
이 때 column 범위 지정 안했을 경우 해당 조건에 해당하는 모든 row의 값이 변경된다.
"""

states.loc[states.density > 100, ['density']] = 10  #density컬럼만 수정
print(states)
print('=======================================')
# states.loc[states.density>100] = 10               #density속성이 100이상인 row를 모두 10으로 수정
# print(states)
print('=======================================')