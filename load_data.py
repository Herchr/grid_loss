import pandas as pd
import clean_data


def load(grid_num):
    df_train = pd.read_csv('train.csv', delimiter=',', parse_dates=True, index_col="Unnamed: 0")
    df_train.dataframeName = 'train.csv'
    df_test = pd.read_csv("test.csv", delimiter=",", parse_dates=True, index_col="Unnamed: 0")
    df_test.dataframeName = 'test.csv'
    x_train, y_train, x_test, y_test = clean_data.clean(df_train, df_test, grid_num)
    return x_train, y_train, x_test, y_test

