import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# Streamlit UI Configuration
# ----------------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("\U0001F4C8 Stock Price Prediction App (LSTM & SVM)")

data_source = st.sidebar.radio("Choose Data Source:", ["Yahoo Finance", "Upload CSV"])
forecast_days = st.sidebar.slider("Days to Forecast", min_value=1, max_value=30, value=7)

# ----------------------------
# Load Data
# ----------------------------
data = None
if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    @st.cache_data
    def load_yahoo_data(ticker, start, end):
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        return df if not df.empty else None

    data = load_yahoo_data(ticker, start_date, end_date)

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Close' columns", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            if "Date" not in df.columns or "Close" not in df.columns:
                st.error("CSV must contain 'Date' and 'Close' columns.")
                st.stop()
            df = df.sort_values("Date")
            data = df[["Date", "Close"]].copy()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

# ----------------------------
# Display & Validate Data
# ----------------------------
if data is None or data.empty:
    st.warning("⚠️ No data available. Please check your input.")
    st.stop()

st.subheader("\U0001F4CA Historical Close Price")
st.line_chart(data.set_index("Date")["Close"])

df = data.copy()

# ----------------------------
# Support Vector Machine Section
# ----------------------------
st.subheader("🔍 SVM Forecast")

df_svm = df.copy()
df_svm["Target"] = df_svm["Close"].shift(-forecast_days)
df_svm.dropna(inplace=True)

X_svm = df_svm[["Close"]]
y_svm = df_svm["Target"]

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

svm_model = SVR()
svm_model.fit(X_train_svm, y_train_svm)
pred_svm = svm_model.predict(X_test_svm)
mse_svm = mean_squared_error(y_test_svm, pred_svm)

future_input = df_svm[["Close"]].tail(forecast_days)
future_pred_svm = svm_model.predict(future_input)
future_dates_svm = pd.date_range(df_svm["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
forecast_svm_df = pd.DataFrame({"Date": future_dates_svm, "Forecast": future_pred_svm})

fig_svm = go.Figure()
fig_svm.add_trace(go.Scatter(x=df_svm["Date"], y=df_svm["Close"], name="Historical"))
fig_svm.add_trace(go.Scatter(x=forecast_svm_df["Date"], y=forecast_svm_df["Forecast"], name="SVM Forecast"))
fig_svm.update_layout(title=f"SVM - MSE: {mse_svm:.2f}", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_svm, use_container_width=True)

# ----------------------------
# LSTM Forecast Section
# ----------------------------
st.subheader("🔮 LSTM Forecast")

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[["Close"]].values)

# Create sequences
window_size = 60

def create_sequences(data, window_size, forecast_days):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_days + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_days].flatten())
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_close, window_size, forecast_days)

split_idx = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(50),
    Dense(forecast_days)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)

last_sequence = scaled_close[-window_size:]
last_sequence = np.expand_dims(last_sequence, axis=0)
forecast_scaled = model.predict(last_sequence)[0]
forecast_lstm = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

future_dates_lstm = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
forecast_lstm_df = pd.DataFrame({"Date": future_dates_lstm, "Forecast": forecast_lstm})

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
fig_lstm.add_trace(go.Scatter(x=forecast_lstm_df["Date"], y=forecast_lstm_df["Forecast"], name="LSTM Forecast"))
fig_lstm.update_layout(title="LSTM Forecast", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_lstm, use_container_width=True)
