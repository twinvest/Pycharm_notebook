from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

"""
1단계 : 데이터 로드
데이터 셋의 형태를 파악하기 위해 데이터를 출력해본다. 
데이터셋의 형태는 딕셔너리 형태로, key가 data와 target으로 구분되어있다. 
친절하게 scikit-learn에서는 데이터를 미리 전처리해두어서 data(X, 독립변수), target(Y, 종속변수)로 구분지어놨다.
그런데, 앞에서 봐왔던 모델들은 data와 target 두 개만이 있었는데 조금더 이상한 것들이 생겼다.
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
target_naems는 타겟의 이름, feature_names은 피쳐의 이름, DESCR은 속성에 대한 디스크립션
"""
data = datasets.load_breast_cancer()
#print(data.keys())
#print(data)
#print(data['DESCR'])
#입력받은 데이터를 X,Y로 구분하여 처리한다. x 데이터의 형태를 확인하기 위해 아래와 같이 진행한다
x = data['data']
y = data['target']
print(x.shape)  #출력결과 569개의 튜플과 30개의 속성이 있는 것을 확인할 수 있다.


"""
2단계 : 데이터분할
데이터 분할은 scikit-learn에서 제공하는 train_test_split을 이용하면 편리하다. 
아래와 같이 split을 하게되면 하나의 데이터로 부터 특정 비율만큼을 트레이닝셋으로, 나머지를 테스트셋으로 구분하여 활용할 수 있다.
"""
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#print(X_train.shape)
#print(X_test.shape)
#print(Y_train.shape)
#print(Y_test.shape)

"""
3단계 : Training Model
scikit-learn에서의 모델 트레이닝은 매우 간단한데, 아래와 같이 특정 모델의 객체를 생성 한 후, 각 객체에 공통적으로 존재하는 fit함수를 사용하면 주어진 데이터에 대해서 학습을 진행한다.
"""
tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)
print(tree)

#모델이 얼마나 fitting이 되었는지 confusion matrix와 accuracy로 확인한다.
print(confusion_matrix(Y_train, tree.predict(X_train)))
print(accuracy_score(Y_train, tree.predict(X_train))) #출력 해보면 학습데이터에 대해서 1이 나온다. 이거 오버피팅임. 왜 오버피팅임? 파라미터튜닝 안함. 즉, 우리는 아무것도 안줬음.
print(confusion_matrix(Y_test, tree.predict(X_test)))
print(accuracy_score(Y_test, tree.predict(X_test))) #TEST어커러시도 90% 가까이 나온다. 이건 데이터셋이 존나 작아서 그럼.

"""
현재 트레이닝이 되어있는 것을 보면 학습 데이터에 대해 정확도가 1이다. 이는 학습 데이터에 대해 과적합되어있다는 의미이다. 왜 이런가..?
이는, 우리가 사용한 decision tree의 파라미터 세팅을 보면 확인 할 수 있다. 현재 아무런 세팅을 하지 않고 진행을 했으므로 모두 default로 설정이 되어있고, 그렇다면 이렇게 과적합이 일어난 이유는 어떤것인지 확인할 필요가 있다.

파라미터 1. max_depth : int or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
파라미터 중 max_depth 옵션을 보면 특정 값으로 설정하지 않을 경우 모든 잎이 pure해질 때까지 분기를 하는 것이다. 
즉, 위에 개념에서 언급한 가지치기가 되어있지 않은 형태이다. 이와 밀접히 관련된 또하나의 파라미터는 min_samples_leaf이다.

min_samples_leaf : int, float, optional (default=1)
The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. 
This may have the effect of smoothing the model, especially in regression.
즉, 우리는 다른 파라미터 뿐만아니라 이 두 가지 파라미터에 대해서 신경써서 학습을 진행해줘야한다.
"""

tree_pruned = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3) #max_depth는 깊이임. default는 None. 이건 끝까지 가겠다임.
#min_samples_leaf는 마지막 노드의 개수. 즉, 불순도가 조금 높아질 수 있으나 일반화를 위해 이를 적당하게 설정해야한다.
#min_samples_leaf는 불순도를 높이고 모델을 일반화 시킨다. 즉, 오버피팅을 방지시킬 수 있게된다.
tree_pruned.fit(X_train, Y_train)
print('===============================================')
print(tree_pruned.score(X_train, Y_train))
#print(accuracy_score(Y_train, tree_pruned.predict(X_train)))
print(tree_pruned.score(X_test, Y_test))
#print(accuracy_score(Y_test, tree_pruned.predict(X_test)))
print('===============================================')

"""
위 코드의 결과를 출력해보면, 학습 데이터의 성능은 떨어졌지만 테스트 데이터의 성능이 높아져 좀 더 일반화가 진행된 트리가 학습됨을 알 수 있다.
추가로, 트리 알고리즘을 활용했을 경우, 모델에서 각 변수 별 가중치 및 중요도를 확인 할 수 있다. 
이는 모델의 속성값인 feature importance 를 이용한다. 이러한 feature importance를 이용하여 트리를 통해 주요 변수를 추출 할 수도 있다.
"""
print(tree_pruned.feature_importances_)
#feature_importances_는 어떤 변수가 중요한지를 보여준다고 생각하면 된다. 피쳐의 개수는 당연히 30개일거임.
#그리고 몇 가지의 속성들은 높은 수치를 보이는데 아래와 같은 막대차트로 시각화 할 수 있고 우린 이걸 참고해서 더 좋은 실험을 할 수 있음.
#또한, 트리 자체도 시각화 할 수 있음. 내가 넣으려했는데 오류 존나 나서 못넣었다. 니들이 한번 찾아봐라.
n_features = X_train.shape[1]
plt.barh(range(n_features), tree_pruned.feature_importances_, align='center')
plt.yticks(np.arange(n_features), data.feature_names)
plt.xlabel("feature importance")
plt.ylabel("feature name")
plt.ylim(-1, n_features)
plt.show()