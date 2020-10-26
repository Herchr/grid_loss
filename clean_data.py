import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def clean(df_train, df_test, grid_num):

    for index, row in df_train.iterrows():
        if row["has incorrect data"]:
            df_train = df_train.drop(index)
            continue
        for col in row:
            if math.isnan(col):
                df_train = df_train.drop(index)  # Temporary solution
                break

    features = [col for col in df_train if col.startswith(f"grid{grid_num}") or not col.startswith("grid")]
    features.remove(f"grid{grid_num}-loss")

    x_train = df_train[features]
    y_train = df_train[f"grid{grid_num}-loss"].to_frame()

    x_test = df_test[features]
    y_test = df_test[f"grid{grid_num}-loss"].to_frame()

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_test = np.asarray(x_test).astype(np.float32)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test
