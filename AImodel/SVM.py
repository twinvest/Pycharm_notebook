from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics

#Data load부분
data = datasets.load_breast_cancer() #데이터 로드
#print(data)
x = data['data'] #독립변수
y = data['target'] #종속변수
print(x.shape)     #(569, 30) 569개 데이터, 속성은 30개. 즉, 30차원이라는 뜻.
print('==================')


#여기서부턴 데이터 분할. train_test_split 함수 사용.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print('X_train.shape: ',X_train.shape)
print(X_train)
print('==================')
print('X_test.shape: ',X_test.shape)
print(X_test)
print('==================')
print(Y_train.shape)
print(Y_test.shape)
print(Y_test)
print('==================')


"""
Training Model
우선 가장 간단한 linear kernel을 이용하여 SVC를 실행시켜보자.
"""
linear_svm = svm.SVC(kernel='linear') #모델생성
linear_svm.fit(X_train, Y_train)  #학습데이터를 집어넣고 모델 학습시키는 중...
print(linear_svm)
print('==================')
Y_pred = linear_svm.predict(X_test) #테스트 데이터에 대해 예측해보기
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred)) #어커러시 출력 해당 모델의 정확도를 확인한다. 해당 데이터는 linear 커널만으로도 충분히 좋은 테스트 정확도를 보인다
print('==================')


"""
다른 모델들도 실험해보자.
돌려보면 알겠지만, rbf나 sigmoid 커널이 오히려 낮게 나온다. 그 이유는 데이터가 너무 간단하기 때문이다.
"""
kernels = ['poly','rbf','sigmoid']
for kernel in kernels:
    # object 생성 및 fitting
    clf = svm.SVC(kernel = kernel, degree=2, gamma='scale')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("kernel : {}, Accuracy:".format(kernel), metrics.accuracy_score(Y_test, Y_pred))
    print("======================================")
print('==================')


"""
마지막으로 커널이 rbf일때, C와 gamma에 따른 정확도의 차이를 확인해보자
참고로 C는 '얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는 정도이다.(soft margin)'
Gamma는 '하나의 데이터 샘플이 영향력을 행사하는 거리이다. 가우시안의 표준편차 역할을 한다고 생각하면 좋다.'
일단, 돌려보면 값이 다 어커런시가 다 동일하게 나오는데 그 이유는 조사한다음에 주석을 다시 달도록 하겠다.
"""
C = [0.1, 1, 10, 100]
Gamma = [0.1,1, 10, 100]
for c in C:
    for gamma in Gamma:
        clf = svm.SVC(kernel = 'rbf', C=c, gamma=gamma) #참고로 linear로 하면 파라미터 gamma는 영향이 없게 된다.
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print("C={} & Gamma={}, Accuracy:".format(c,gamma), metrics.accuracy_score(Y_test, Y_pred))
        print("======================================")
print('==================')