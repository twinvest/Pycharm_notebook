from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
#matplotlib.use('Agg')
warnings.filterwarnings(action='ignore')
"""
def make_preprocessor():
    base = pd.read_csv('/home/rlaxodn031/tw_python/data2/base1.csv', encoding='euc-kr')
    one = pd.read_csv('/home/rlaxodn031/tw_python/data2/courier.csv', encoding='euc-kr')  # 완료
    two = pd.read_csv('/home/rlaxodn031/tw_python/data2/languagexam.csv', encoding='euc-kr')  # 완료
    three = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_age.csv', encoding='euc-kr')  # 완료
    four = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_area_range.csv', encoding='euc-kr')  # 완료
    five = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_children_count.csv', encoding='euc-kr')  # 완료
    six = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_club_count.csv', encoding='euc-kr')  # 완료
    seven = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_gender.csv', encoding='euc-kr')  # 완료
    eight = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_honey_YN.csv', encoding='euc-kr')  # 완료
    nine = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_religion.csv', encoding='euc-kr')  # 완료
    ten = pd.read_csv('/home/rlaxodn031/tw_python/data2/retire.csv', encoding='euc-kr')  # 완료
    eleven = pd.read_csv('/home/rlaxodn031/tw_python/data2/workingyear.csv', encoding='euc-kr')  # 완료
    twelve = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_position.csv', encoding='euc-kr')  # 완료
    thirteen = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_staff_division.csv', encoding='euc-kr')  # 완료

    fourteen = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_department.csv', encoding='utf-8')
    fifteen = pd.read_csv('/home/rlaxodn031/tw_python/data2/new_merge_academic_ability.csv', encoding='utf-8')
    sixteen=pd.read_csv('/home/rlaxodn031/tw_python/data2/certificate.csv', encoding='utf-8')
    seventeen=pd.read_csv('/home/rlaxodn031/tw_python/data2/duration.csv', encoding='utf-8')
    eighteen=pd.read_csv('/home/rlaxodn031/tw_python/data2/fired.csv', encoding='utf-8')
    twentyone=pd.read_csv('/home/rlaxodn031/tw_python/data2/univ_duration.csv', encoding='utf-8')
    return base, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, twentyone
"""
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



def make_file(df):
    #path='/home/rlaxodn031/tw_python/xgb/result.csv'
    path = './result.csv'
    df.to_csv(path, encoding="ms949", mode='w', index=False)

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
    return one, two, eight, nine, eleven

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


    #print(df.shape)
    df=df.drop('fired', axis=1)

    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #결측값 제거 코드
    #make_file(df)
    #print(df.shape)
    #print('외국인 체크:', df['foreigner'].mean())
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

    #print('정규직 체크:', df['emp_status'].mean())

    df = df.drop('emp_status', axis=1)

    #print(df.loc[df['retirement']==1,'working_year'].quantile([0.25, 0.5, 0.75, 1.0]))
    #print(df.loc[df['retirement'] == 1, 'working_year'].quantile(np.linspace(0,1,4)))


    df['re_class']=0
    df.loc[df['working_year']>=8, 're_class']=1
    #print(df['re_class'].groupby(df['re_class']).count())

    # print(df.loc[df['retirement']==1,'working_year'].quantile([0.2, 0.4, 0.6, 0.8, 1.0]))
    df['re_class2']=None

    df.loc[df['working_year']<=1,'re_class2']=0
    df.loc[(df['working_year']>1) & (df['working_year']<=3), 're_class2']=1
    df.loc[(df['working_year']>3) & (df['working_year']<=4), 're_class2']=2
    df.loc[(df['working_year']>4), 're_class2']=3

    #print(df['re_class2'].groupby(df['re_class2']).count())


    df['re_class3']=None

    df.loc[df['working_year']<=1, 're_class3']=0
    df.loc[(df['working_year']>1) & (df['working_year']<=3), 're_class3']=1
    df.loc[(df['working_year']>3), 're_class3']=2
    #print(df['re_class3'].groupby(df['re_class3']).count())




    #df0=df.loc[df['retirement']==0]
    #df1=df.loc[df['retirement']==1]

    #describe0=pd.DataFrame(df0.describe())
    #describe1=pd.DataFrame(df1.describe())

    #path = '/home/rlaxodn031/tw_python/xgb/describe0.csv'
    #describe0.to_csv(path, mode="w", index=True)

    #path = '/home/rlaxodn031/tw_python/xgb/describe1.csv'
    #describe1.to_csv(path, mode="w", index=True)


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
    #plt.savefig('/var/lib/tomcat8/webapps/Test/xgb/tw.png')
    plt.savefig('./pic/tw.png')
    return ax

def model_save(model):
    import pickle
    pickle.dump(model, open("pima.pickle.dat", "wb"))

def model_open():
    import pickle
    loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
    return loaded_model

def predict_by_save_model():
    print("###################################")
    print("자 이제 저장된 모델을 불러올 것입니다.")
    print("###################################")
    model = model_open()
    raw_data = {'gender': [1],
                'honey_YN': [0],
                'child_count': [0],
                'club': [0],
                'courier_count': [0],
                'main_dep': [0],
                'age': [29],
                'certificate': [0],
                'duration': [0.409538611111111],
                'position_number_1.0': [0],
                'position_number_2.0': [0],
                'position_number_3.0': [0],
                'position_number_4.0': [0],
                'position_number_5.0': [0],
                'position_number_6.0': [1],
                'area_Chungbuk': [0],
                'area_Chungnam': [0],
                'area_Gangwon': [0],
                'area_Jeonam': [0],
                'area_Jeonbuk': [0],
                'area_Kyongbuk': [0],
                'area_Kyongnam': [1],
                'area_Metropolitan': [0],
                'department_Business_Support': [0],
                'department_Business_management': [0],
                'department_Design': [0],
                'department_Financial_management': [0],
                'department_Production_Management': [0],
                'department_Production_support': [0],
                'department_Purchasing': [0],
                'department_Quality_Management': [0],
                'department_Research': [0],
                'department_Sales': [0],
                'department_Service': [0],
                'department_Shipping': [0],
                'department_Technology': [0],
                'department_Warranty': [1]
                }
    data = pd.DataFrame(raw_data)
    print(data)
    X_TEST = data
    print("저장된 모델1", model.predict(X_TEST))
    print("저장된 모델2", model.predict_proba(X_TEST))


def XGBclass2(df):
    from sklearn.metrics import f1_score
    from sklearn import metrics
    import numpy as np
    import xgboost as xgb
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    
    #df=df.loc[df['retirement']==1]
    y = df['re_class']  # y는 종속변수
    # name = ['gender', 'honey_YN', 'position_number', 'area', 'child_count', 'staff_type', 'club', 'working_year']
    name = ['gender','honey_YN','position_number','area','child_count','staff_type','club','courier_count','main_dep','department','age','certificate','duration']
    x = df.loc[:, name]  # x는 독립변수

    x = pd.get_dummies(x, columns=['position_number'])  # 직책 one-hot encoding
    x = pd.get_dummies(x, columns=['area'])  # 지역 one-hot encoding
    x = pd.get_dummies(x, columns=['staff_type'])  # 정규직, 비정규직, 촉탁직 one-hot encoding
    x = pd.get_dummies(x, columns=['department'])
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)  # 학습데이터와 테스트데이터를 쪼갬
    print("shape of train_set/test_set")
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


    """
    max model
    """
    xg_reg = xgb.XGBClassifier(learning_rate=0.7, max_depth=3, n_estimators=30, gamma=0, min_child_weight=0, probability=True)
    xg_reg.fit(X_train, Y_train)

    make_file(X_test)
    Y_pred = xg_reg.predict(X_test)
    Y_proba1 = xg_reg.predict_proba(X_test)


    print(X_test.shape)
    print('태우쓰예측1', Y_pred)
    print("태우쓰확률2", Y_proba1)


    class_name=np.array(['8year less', '8year over'])
    print(plot_confusion_matrix(Y_test, Y_pred, classes=class_name))
    xgb_acc = metrics.accuracy_score(Y_test, Y_pred)


    #####################
    skplt.metrics.plot_cumulative_gain(Y_test, Y_proba1)
    plt.savefig('./pic/xgb_class2_gain.png')
    plt.cla

    """
    plot tree
    """
    from xgboost import plot_tree
    plot_tree(xg_reg)
    plt.savefig('./pic/xgb_class2_tree.png')
    plt.show()
    plt.close()

    ######################
    """
    feature importance
    """
    #######################
    n_features = X_train.shape[1]
    plt.figure(figsize=(8, 12))
    plt.subplots_adjust(left=0.25)
    plt.barh(range(n_features), xg_reg.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x)
    plt.xlabel("feature importance, Accuracy: %s" %xgb_acc)
    plt.ylabel("feature name")
    plt.ylim(-1, n_features)
    #plt.savefig('/var/lib/tomcat8/webapps/Test/xgb/xgb_class2_fimp.png')
    plt.savefig('./pic/xgb_class2_fimp.png')
    plt.show()
    model_save(xg_reg)



def main():
    base, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen,twentyone = \
        make_preprocessor() #전처리
    id = base['직원id']
    foreigner = base['내외국인']
    base1 = make_dataframe(id, foreigner) #base1만들기
    one, two, eight, nine, eleven = change_columnname(one, two, eight, nine, eleven, twentyone) #column_name변경
    df = make_merge(base1, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen,twentyone) #merge와 결측치처리 함수. 반환값은 모든 처리가 완료된 dataframe.

    # 장기근속 모형  95%
    #print('XGBClass, working year > 8')
    XGBclass2(df)
    predict_by_save_model() #save된 모델 호출 함수


if __name__ == "__main__":
    main()