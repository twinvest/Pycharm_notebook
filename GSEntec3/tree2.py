import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib

matplotlib.use('Agg')  # linux에서 쓰려면 이거 필요


def make_preprocessor():
    base = pd.read_csv('./data/base1.csv', encoding='euc-kr') #완료
    one = pd.read_csv('./data/courier.csv')  # 완료
    two = pd.read_csv('./data/languagexam.csv')  # 완료
    three = pd.read_csv('./data/new_age.csv')  # 완료
    four = pd.read_csv('./data/new_area_range.csv', encoding='euc-kr')  # 완료
    five = pd.read_csv('./data/new_children_count.csv', encoding='euc-kr')  # 완료
    six = pd.read_csv('./data/new_club_count.csv')  # 완료
    seven = pd.read_csv('./data/new_gender.csv')  # 완료
    eight = pd.read_csv('./data/new_honey_YN.csv', encoding='euc-kr')  # 완료
    nine = pd.read_csv('./data/new_religion.csv')  # 완료
    ten = pd.read_csv('./data/retire.csv')  # 완료
    eleven = pd.read_csv('./data/workingyear.csv', encoding='euc-kr')  # 완료
    twelve = pd.read_csv('./data/new_position.csv', encoding='euc-kr')  # 완료
    thirteen = pd.read_csv('./data/new_staff_division.csv', encoding='euc-kr')  # 완료
    fourteen = pd.read_csv('./data/new_department.csv', encoding='euc-kr')
    fifteen = pd.read_csv('./data/new_merge_academic_ability.csv', encoding='UTF-8')
    sixteen=pd.read_csv('./data/certificate.csv', encoding='euc-kr')
    seventeen=pd.read_csv('./data/duration.csv', encoding='euc-kr')
    eighteen=pd.read_csv('./data/fired.csv', encoding='euc-kr')
    twentyone=pd.read_csv('./data/univ_duration.csv', encoding='euc-kr')

    return base, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, twentyone

def make_dataframe(id, foreigner):
    base1 = pd.DataFrame({'id': id,
                       'foreigner' : foreigner
                       })
    base1.loc[base1['foreigner']=='내국인', 'foreigner'] =1
    base1.loc[base1['foreigner']=='외국인', 'foreigner'] =0

    return base1

def change_columnname(one, two, eight, nine, eleven, twentyone):
    one.rename(
        columns={'count':'courier_count'}, inplace=True
    )
    two.rename(
        columns={'count':'lang_count'}, inplace=True
    )
    eight.rename(
        columns={'relations':'relation_spouse'}, inplace=True
    )
    nine.rename(
        columns={'exist_nonexist':'religion_YN'}, inplace=True
    )
    eleven.rename(
        columns={'직원id':'id',
                 '입사일':'begin_date',
                 '퇴사일':'retired_date',
                 '근속연수(연)' : 'working_year',
                 '근속연수(월)' : 'working_month',
                 '근속연수(일)' : 'working_day'
                 }, inplace=True
    )

    twentyone.rename(
        columns={'duration': 'univ_duration'}, inplace=True
    )


    return one, two, eight, nine, eleven, twentyone


def make_merge(base1, one, two, three, four, five, six, seven, eight, nine, ten,
               eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, twentyone):
    import numpy as np
    df = base1.merge(ten, on='id', how='left')  # ten은 퇴사여부
    df = df.merge(one, on='id', how='left') #커리어 갯수
    df = df.merge(two, on='id', how='left') #외국어 시험
    df = df.merge(three, on='id', how='left') # 연령
    df = df.merge(four, on='id', how='left') #지역범주
    df = df.merge(five, on='id', how='left') #아이의 수. left join해줘야 한다. 여기선!
    df = df.merge(six, on='id', how='left') #동호회 가입
    df = df.merge(seven, on='id', how='left') #seven은 성별
    df = df.merge(eight, on='id', how='left') #eight은 결혼여부. 이거 결혼한 애들 번호밖에 없음. 따라서 outer조인 후 결측값 생긴 부분 0으로 바꿔줘야 한다.
    df = df.merge(nine, on='id', how='left') # 종교
    df = df.merge(eleven, on='id', how='left') # 근속연수
    df = df.merge(twelve, on='id', how='left') #twelve는 직책. 여기서 외국인이나 임원들은 전부 결측치인데 이거 다 제거할 것!
    df = df.merge(thirteen, on='id', how='left')
    df = df.merge(fourteen, on='id', how='left')
    df = df.merge(fifteen, on='id', how='left')
    df = df.merge(sixteen, on='id', how='left')
    df = df.merge(seventeen, on='id', how='left')
    df = df.merge(eighteen, on='id', how='left')
    df = df.merge(twentyone, on='id', how='left')




    # 한글 컬럼 드랍
    df = df.drop('position', axis=1)
    df = df.drop('relations', axis=1)
    df = df.drop('relation_spouse', axis=1)
    df = df.drop('employment_text', axis=1)
    df = df.drop('gender_text', axis=1)
    df = df.drop('school_name', axis=1)
    df = df.drop('graduate', axis=1)
    df = df.drop('kinds', axis=1)
    df = df.drop('lat', axis=1)
    df = df.drop('lng', axis=1)

    # #종교 유무 코딩
    # df.loc[:,'religion_bin']==0
    # df.loc[df['religion_YN']=='Yes', 'religion_YN' ]=1

    # 결측치처리
    missing = {'honey_YN': 0,
                'child_count' : 0,
                'club_count': 0,
                'club': 0,
                'courier_count': 0,
                'lang_count': 0,
                'certificate': 0,
                }
    df = df.fillna(missing)
    df = df.drop('Unnamed: 2', axis=1)
    df = df.drop(df[df['fired']==1].index, axis=0)
    print(df.shape)
    df=df.drop('fired', axis=1)

    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #결측값 제거 코드
    print(df.shape)

    print('외국인 체크:', df['foreigner'].mean())
    df = df.drop('foreigner', axis=1)



    df.rename(
        columns={
            'result':'emp_status'
        }, inplace= True
    )

    df.rename(
        columns={
            'employment':'retirement'
        }, inplace= True
    )

    df.loc[df['emp_status']=='정규직', 'emp_status']=1
    df.loc[df['emp_status']=='촉탁직', 'emp_status']=2
    df.loc[df['emp_status'] == '비정규직', 'emp_status']=3

    df['duration']=df['duration']/3600000

    print('정규직 체크:', df['emp_status'].mean())

    df = df.drop('emp_status', axis=1)

    print(df.loc[df['retirement']==1,'working_year'].quantile([0.25, 0.5, 0.75, 1.0]))

    print(df.loc[df['retirement'] == 1, 'working_year'].quantile(np.linspace(0,1,4)))



    df['re_class']=0

    df.loc[df['working_year']>=8, 're_class']=1

    print(df['re_class'].groupby([df['re_class'], df['retirement']]).count())



    # print(df.loc[df['retirement']==1,'working_year'].quantile([0.2, 0.4, 0.6, 0.8, 1.0]))
    #
    #
    #
    df['re_class2']=None

    df.loc[df['working_year']<=1,'re_class2']=0
    df.loc[(df['working_year']>1) & (df['working_year']<=3), 're_class2']=1
    df.loc[(df['working_year']>3) & (df['working_year']<=4), 're_class2']=2
    df.loc[(df['working_year']>4), 're_class2']=3

    print(df['re_class2'].groupby(df['re_class2']).count())


    df['re_class3']=None

    df.loc[df['working_year']<=1, 're_class3']=0
    df.loc[(df['working_year']>1) & (df['working_year']<=3), 're_class3']=1
    df.loc[(df['working_year']>3), 're_class3']=2
    print(df['re_class3'].groupby(df['re_class3']).count())




    df0=df.loc[df['retirement']==0]
    df1=df.loc[df['retirement']==1]

    describe0=pd.DataFrame(df0.describe())
    describe1=pd.DataFrame(df1.describe())

    path = './describe0.csv'
    describe0.to_csv(path, mode="w", index=True)

    path = './describe1.csv'
    describe1.to_csv(path, mode="w", index=True)

    df.to_csv('./dataframe_merge.csv', mode= 'w', index=True)


    return df


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          ):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def print_numOfNan(df):
    num1 = df['id'].isnull().sum()
    num2 = df['department'].isnull().sum()
    num3 = df['foreigner'].isnull().sum()
    num4 = df['employment'].isnull().sum()
    num5 = df['gender'].isnull().sum()
    num6 = df['honey_YN'].isnull().sum()
    num7 = df['position_number'].isnull().sum()
    num8 = df['child_count'].isnull().sum()

    print('id : ', num1)
    print('department : ', num2)
    print('foreigner : ', num3)
    print('employment : ', num4)
    print('gender : ', num5)
    print('honey_YN', num6)
    print('position_number', num7)
    print('child_count', num8)

    print(df)  # df출력
    print(df.keys())


def make_file(df):
    path = './result2.csv'
    df.to_csv(path, encoding="ms949", mode="w", index=False)  # 파일생성


def make_pictureofTree2(tree_pruned, x):
    #트리그리기
    from sklearn.tree import export_graphviz
    export_graphviz(tree_pruned, feature_names=x.columns.tolist(), out_file='tree.dot')
    # from subprocess import call
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    import matplotlib.pyplot as plt
    plt.figure(figsize = (14, 18), dpi=1000)
    plt.imshow(plt.imread('./pic/tree.png'))
    plt.axis('off')
    plt.show()



def make_pictureofImportance(X_train, tree_pruned, x):
    import matplotlib.pyplot as plt
    import numpy as np

    n_features = X_train.shape[1]
    plt.figure(figsize=(15,8))
    plt.subplots_adjust(left=0.25)
    plt.barh(range(n_features), tree_pruned.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x)
    plt.xlabel("feature importance")
    plt.ylabel("feature name")
    plt.ylim(-1, n_features)


    plt.savefig('f_imp')
    plt.show()


def DTree2(df):
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, accuracy_score

    # df = df.loc[df['retirement'] == 1]

    y = df['re_class']  # y는 종속변수
    # name = ['gender', 'honey_YN', 'position_number', 'area', 'child_count', 'staff_type', 'club', 'working_year']
    name = ['gender',
            'honey_YN',
            'position_number',
            'area',
            'child_count',
            'staff_type',
            'club',
            'courier_count',
            'main_dep',
            'department',
            'age',
            'certificate',
            'duration'

            ]
    x = df.loc[:, name]  # x는 독립변수

    x = pd.get_dummies(x, columns=['position_number'])  # 직책 one-hot encoding
    x = pd.get_dummies(x, columns=['area'])  # 지역 one-hot encoding
    x = pd.get_dummies(x, columns=['child_count'])  # 애색히 one-hot encoding
    x = pd.get_dummies(x, columns=['staff_type'])  # 정규직, 비정규직, 촉탁직 one-hot encoding
    x = pd.get_dummies(x, columns=['department'])
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)  # 학습데이터와 테스트데이터를 쪼갬
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    """
    decision tree model
    """
    tree_pruned = DecisionTreeClassifier(max_depth=7,
                                         min_samples_leaf=2
                                         )

    tree_pruned.fit(X_train, Y_train)
    Y_pred=tree_pruned.predict(X_test)
    Y_proba = tree_pruned.predict_proba(X_test)

    class_name = np.array(['less_8year', 'over_8year'])

    print(plot_confusion_matrix(Y_pred, Y_test, class_name))

    ##################################################################################################
    """
    트리 그림 그리기
    """
    make_pictureofTree2(tree_pruned, x)
    ##################################################################################################
    """
    속성중요도 그리기
    """
    make_pictureofImportance(X_train, tree_pruned, x)

    ###################################################################################################



def main():
    base, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen,twentyone = make_preprocessor() #전처리
    id = base['직원id']
    foreigner = base['내외국인']
    base1 = make_dataframe(id, foreigner) #base1만들기
    one, two, eight, nine, eleven, twentyone = change_columnname(one, two, eight, nine, eleven, twentyone) #column_name변경
    df = make_merge(base1, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen,twentyone) #merge와 결측치처리 함수. 반환값은 모든 처리가 완료된 dataframe.
    DTree2(df)



if __name__ == "__main__":
    main()