import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import yfinance as yf
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

with open('Model/built_models/lr_tts_google.pkl', 'rb') as file:
    lrg = pickle.load(file)

with open('Model/built_models/lr_tts_nabil.pkl', 'rb') as file:
    lrn = pickle.load(file)

with open('Model/built_models/svm_tts_google.pkl', 'rb') as file:
    svmg = pickle.load(file)

with open('Model/built_models/svm_tts_nabil.pkl', 'rb') as file:
    svmn = pickle.load(file)

with open('Model/built_models/rf_tts_google.pkl', 'rb') as file:
    rfg = pickle.load(file)

with open('Model/built_models/rf_tts_nabil.pkl', 'rb') as file:
    rfn = pickle.load(file)

cnn_google = load_model('Model/built_models/cnn_model_train_test_split.h5')
cnnlstm_google = load_model('Model/built_models/cnnlstm_tts_google.h5')
lstm_google = load_model('Model/built_models/lstm_tts_google.h5')
cnnlstm_nabil = load_model('Model/built_models/cnnlstm_tts_nabil.h5')
cnn_nabil = load_model('Model/built_models/cnn_tts_nabil.h5')
lstm_nabil = load_model('Model/built_models/lstm_tts_nabil.h5')

# Placeholder functions to simulate data and results.
def load_dataset(stock):
    # Simulate loading a dataset
    if stock == 'NABIL':
        data = pd.read_csv(f'Dataset/{stock}.csv', index_col=0)
    else:
        data = pd.read_csv(f'Dataset/{stock}.csv')
    return data

def get_performance_metrics(stock):
    # Simulate performance metrics
    if stock == 'GOOGL':
        metrics = pd.DataFrame({
            # Define the data
            'Algorithm': ['LR', 'SVM', 'RF', 'XGB', 'CNN', 'LSTM', 'CNN-LSTM', 'SARIMAX', 'FB-Prophet'],
            'MSE (Train Test Split)': [0.168, 0.167, 0.415, 0.614, 6.372, 1.879, 11.416, 0.846, 193.277],
            'MSE (First 80% Train)': [0.653, 0.660, 2365.704, 2773.901, 197.659, 34.076, 2234.370, 9.485, 193.277],
            'RMSE (Train Test Split)': [0.410, 0.409, 0.644, 0.784, 2.524, 1.371, 3.378, 0.920, 13.902],
            'RMSE (First 80% Train)': [0.808, 0.812, 48.638, 52.667, 14.059, 5.837, 47.269, 3.079, 13.902],
            'MAE (Train Test Split)': [0.218, 0.217, 0.325, 0.374, 1.747, 0.741, 2.853, 0.768, 10.923],
            'MAE (First 80% Train)': [0.626, 0.630, 41.600, 46.262, 13.114, 4.745, 40.717, 2.689, 10.923],
            'Training Time (sec)': [0.023, 1.283, 2.320, 2.679, 131.116, 1934.98, 201.006, 30.902, 6.552]
            })
    else:
        metrics = pd.DataFrame({
            # Define the data
            'Algorithm': ['LR', 'SVM', 'RF', 'XGB', 'CNN', 'LSTM', 'CNN-LSTM', 'SARIMAX', 'FB-Prophet'],
            'MSE (Train Test Split)': [176.76, 179.88, 254.34, 282.95, 2417.73, 1863.13, 7682.09, 12919.91, 10493.07],
            'MSE (First 80% Train)': [32.13, 290.18, 10885.80, 11862.64, 38259.15, 1984.71, 17683.82, 14843.97, None],
            'RMSE (Train Test Split)': [13.10, 13.411, 15.94, 16.82, 49.17, 43.16, 87.64, 113.66, 102.43],
            'RMSE (First 80% Train)': [5.66, 17.03, 104.33, 108.91, 195.59, 44.55, 132.95, 121.84, None],
            'MAE (Train Test Split)': [8.76, 8.89, 9.84, 10.66, 34.01, 24.66, 58.81, 99.69, 86.13],
            'MAE (First 80% Train)': [4.17, 12.22, 71.79, 76.59, 150.12, 39.87, 111.23, 104.39, None],
            'Training Time (sec)': [0.031, 1.001, 2.255, 1.234, 91.938, 966.430, 135.355, 93.043, 6.603]
            })
    return metrics

def get_basic_ml_model(stock):
    # Simulate predictions
    if stock == 'GOOGL':
        model = {
            'Linear Regression': lrg,
            'SVM': svmg,
            'Random Forest': rfg
        }
    else:
        model = {
            'Linear Regression': lrn,
            'SVM': svmn,
            'Random Forest': rfn
        }
    return model

def get_dl_model(stock):
    # Simulate predictions
    if stock == 'NABIL':
        model = {
            'CNN': cnn_nabil,
            'LSTM': lstm_nabil,
            'CNN-LSTM': cnnlstm_nabil
        }
    else:
        model = {
            'CNN': cnn_google,
            'LSTM': lstm_google,
            'CNN-LSTM': cnnlstm_google
        }
    return model

# Streamlit App
st.title("Performance Evaluation and Analysis of Machine Learning Algorithms in Stock Price Prediction")

# Dropdown for stock selection
stocks = ['GOOGL', 'NABIL']
selected_stock = st.selectbox("Select a stock:", stocks)

# Load data
data = load_dataset(selected_stock)

st.subheader(f"Dataset for {selected_stock}")
st.dataframe(data)

# Line graph of closing prices
st.subheader(f"Closing Price Line Graph for {selected_stock} Stock Dataset")
fig_line = plt.figure()
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Closing Prices of {selected_stock}')
plt.legend()
st.pyplot(fig_line)

# Candlestick chart
st.subheader(f"Candlestick Chart for {selected_stock}")
fig_candle = go.Figure(data=[
    go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close']
    )
])
fig_candle.update_layout(title=f'Candlestick Chart for {selected_stock}', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig_candle)

# Performance evaluation
st.subheader("Performance Evaluation Metrics")
performance_metrics = get_performance_metrics(selected_stock)
st.dataframe(performance_metrics)

# Simulate basic machine learning predictions
data = load_dataset(selected_stock)
data['Date'] = pd.to_datetime(data['Date'])
X = data.drop(['Close', 'Date'], axis=1)
y = data['Close']
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Predictions and their line graphs
st.subheader("Prediction Line Graphs")
models = get_basic_ml_model(selected_stock)

for algorithm, model in models.items():
    st.write(f"{algorithm} Predictions")
    predicted = model.predict(X_test)
    prediction = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
    for a in prediction.index:
        for b in data.index:
            if a == b:
                prediction['Date'] = data['Date']
    prediction.set_index('Date', inplace=True)
    prediction.sort_index(inplace=True)
    st.subheader(f"Prediction for {selected_stock}")
    st.dataframe(prediction)
    fig_pred = plt.figure()
    plt.plot(prediction.Actual, label='Actual Value', color='g', linewidth=1)
    plt.plot(prediction.Predicted, label=f'{algorithm} Prediction', color='r', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.title(f'{algorithm} Predictions for {selected_stock}')
    plt.legend()
    st.pyplot(fig_pred)


# Simulate deep learning predictions
data = load_dataset(selected_stock)
data['Date'] = pd.to_datetime(data['Date'])
features = data[['Open', 'High', 'Low', 'Volume']]
target = data[['Close']]
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Scale the features (Open, High, Low, Volume)
scaled_features = scaler_features.fit_transform(features)

# Scale the target (Close)
scaled_target = scaler_target.fit_transform(target)
# Combine the scaled features and scaled target into a new DataFrame
scaled_data = pd.DataFrame(scaled_features, columns=['Open', 'High', 'Low', 'Volume'])
scaled_data['Close'] = scaled_target  # Use the scaled Close values as the target

sequence_length = 30
X = []
y = []
target_index = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[['Open', 'High', 'Low', 'Volume']].values[i-sequence_length:i])
    y.append(scaled_data['Close'].values[i])
    target_index.append(i)  # store index of each target value

# Convert to numpy arrays
X, y = np.array(X), np.array(y)

X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, y_test_indices = train_test_split(target_index, test_size=0.2, random_state=42)

# Predictions and their line graphs
st.subheader("Prediction Line Graphs")
models = get_dl_model(selected_stock)

for algorithm, model in models.items():
    st.write(f"{algorithm} Predictions")
    predictions = model.predict(X_test)
    y_pred_inv = scaler_target.inverse_transform(predictions)
    y_test_inv = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    prediction = pd.DataFrame({'Actual': y_test_inv.flatten(), 'Predicted': y_pred_inv.flatten()}, index=y_test_indices)
    for a in prediction.index:
        for b in data.index:
            if a == b:
                prediction['Date'] = data['Date']
    prediction.set_index('Date', inplace=True)
    prediction.sort_index(inplace=True)
    st.subheader(f"Prediction for {selected_stock}")
    st.dataframe(prediction)
    fig_pred = plt.figure()
    plt.plot(prediction.Actual, label='Actual Value', color='g', linewidth=1)
    plt.plot(prediction.Predicted, label=f'{algorithm} Prediction', color='r', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.title(f'{algorithm} Predictions for {selected_stock}')
    plt.legend()
    st.pyplot(fig_pred)