import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_lstm(series, steps=5, epochs=50, batch_size=1):
    series = np.array(series)
    min_val = np.min(series)
    max_val = np.max(series)
    series_norm = (series - min_val) / (max_val - min_val)

    X, y = [], []
    for i in range(len(series_norm)-1):
        X.append([series_norm[i]])
        y.append(series_norm[i+1])
    X = np.array(X).reshape(-1, 1, 1)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    input_seq = X[-1]
    future = []
    for _ in range(steps):
        pred = model.predict(input_seq.reshape(1,1,1), verbose=0)
        future.append(pred[0,0] * (max_val - min_val) + min_val)
        input_seq = np.array([[pred[0,0]]])

    y_pred = model.predict(X, verbose=0).flatten() * (max_val - min_val) + min_val
    y_true = series[1:]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return future, mse, rmse, mae, model
