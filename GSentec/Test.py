import pandas as pd

def gender_define(dt):
    nullofgendeer = dt['성별구분'].isnull().sum()
    if(nullofgendeer == 0):
        print('결측치가 존재하지 않습니다.')
    else:
        print('결측치가 존재합니다.', nullofgendeer)
        dt2=dt.dropna(subset = ['성별구분']) #성별구분이라는 컬럼에 NaN값이 있으면 해당 행을 삭제

        dt2.loc[dt2['성별구분'] =='남자', '성별구분'] = 1
        dt2.loc[dt2['성별구분'] == '여자', '성별구분'] = 2

        # dt2.loc[dt2.성별구분 =='남자', ['성별구분']] = 1 #남은 1로 대체
        # dt2.loc[dt2.성별구분 == '여자', ['성별구분']] = 2  # 여는 2로 대체
        print(dt2['성별구분'])
        numofman = sum(dt2['성별구분'] == 1)
        numofwoman = sum(dt2['성별구분'] == 2)
        # numofquestion = sum(dt2['성별구분'] != 2 & dt2['성별구분'] != 1)
        nullofgendeer2 = dt2['성별구분'].isnull().sum()
        if(nullofgendeer2 == 0):
            print('이제 결측치가 없습니다.')
        print('남자', numofman)
        print('여자', numofwoman)
        # print('물음표', numofquestion)
        print('총합', numofman + numofwoman)
        print(dt2)
        return dt2

df1 = pd.read_csv("./employee.csv", encoding='euc-kr')
df2 = pd.read_csv("./education.csv", encoding='euc-kr')
df3 = pd.merge(df1, df2, how='outer')
# print(df1)
# print(df2)
# print(df3)

df1 = gender_define(df1) #남자 여자 define


# nullofmajor = df3['학력'].isnull().sum()
# department = df1['부서'].isnull().sum()
#
# print('부서결측치', department)
# print('학력결측치', nullofmajor)