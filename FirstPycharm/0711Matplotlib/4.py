import pandas as pd
def read_dataset():
    return pd.read_csv('iris.csv'), pd.read_csv('iris_metadata.csv')

def merge_dfs(iris, metadata):
    df_merge = pd.merge(iris, metadata, left_on = 'species', right_on = 'name', how='inner')
    return df_merge


"""
이 때 기준 index 값보다 작은 index 는 학습용 dataframe 으로, index 값보다 같거나 큰 index 는 테스트용 dataframe 으로 분할되도록 해야 함 
"""
def split_df(df, ratio):
    splited_index = round(len(df) * ratio)
    #print(splited_index)
    # print(df)
    traing_set = df[:splited_index]
    test_set = df[splited_index:]
    print(traing_set)
    print(test_set)
    return traing_set, test_set

def truncate_non_toxics(df):
    return df.loc[df['toxic']>0]

def main():
    split_ratio = 0.7
    iris, metadata = read_dataset()
    # print(iris)
    # print(metadata)
    merge = merge_dfs(iris, metadata)
    split = split_df(merge, split_ratio)

    test = truncate_non_toxics(split[0])
    train = truncate_non_toxics(split[1])

    print(test.describe())
    print(train.describe())

main()