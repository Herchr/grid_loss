from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
import load_data


grid_num = 1
X_train, y_train, X_test, y_test = load_data.load(grid_num)


model = Sequential()
model.add(Bidirectional(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dense(units=1, activation="relu"))
adam = Adam(learning_rate=0.1)
model.compile(optimizer=adam, loss="mse", )
model.fit(X_train, y_train, epochs=15, validation_split=0.1)

print(model.evaluate(X_test, y_test))
