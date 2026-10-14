import random
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

st.set_page_config(page_title="Dự báo giá cổ phiếu bằng LSTM", layout="wide")
st.title("📈 Dự báo giá cổ phiếu bằng LSTM")

# ---------------- Sidebar ----------------
st.sidebar.header("Cấu hình")
ticker = st.sidebar.text_input("Mã cổ phiếu (Yahoo Finance)", "VNM.VN")
start_date = st.sidebar.date_input("Từ ngày", date(2018, 1, 1))
window_size = st.sidebar.slider("Số ngày nhìn lại (window)", 20, 120, 60, 10)
epochs = st.sidebar.slider("Epochs", 5, 50, 10)
n_future = st.sidebar.slider("Số ngày dự báo", 1, 30, 10)
seed = st.sidebar.number_input("Seed", value=42, step=1)
run = st.sidebar.button("🚀 Huấn luyện & dự báo", type="primary")


# ---------------- Hàm hỗ trợ ----------------
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data(ticker, start):
    df = yf.download(ticker, start=str(start), end=str(date.today()), progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # yfinance bản mới trả về MultiIndex
        df.columns = df.columns.get_level_values(0)
    return df[["Close"]].dropna()


def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def set_seed(s):
    np.random.seed(s)
    tf.random.set_seed(s)
    random.seed(s)


# ---------------- Dữ liệu ----------------
df = load_data(ticker, start_date)
if df.empty:
    st.error("Không tải được dữ liệu. Kiểm tra lại mã cổ phiếu.")
    st.stop()

st.subheader(f"Giá đóng cửa {ticker}")
st.line_chart(df["Close"])

if not run:
    st.info("Chọn cấu hình ở thanh bên rồi bấm **Huấn luyện & dự báo**.")
    st.stop()

if len(df) < window_size * 5:
    st.error("Dữ liệu quá ít so với window. Hãy chọn ngày bắt đầu sớm hơn.")
    st.stop()

# ---------------- Tiền xử lý ----------------
set_seed(int(seed))
data = df[["Close"]].values
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

X_train, y_train = create_sequences(train_scaled, window_size)
test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
X_test, y_test = create_sequences(test_combined, window_size)

X_train = X_train.reshape((X_train.shape[0], window_size, 1))
X_test = X_test.reshape((X_test.shape[0], window_size, 1))

val_size = int(0.1 * len(X_train))
X_val, y_val = X_train[-val_size:], y_train[-val_size:]
X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]

# ---------------- Mô hình ----------------
model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1),
])
model.compile(optimizer="adam", loss="mean_squared_error")

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

with st.spinner("Đang huấn luyện mô hình..."):
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

# ---------------- Đánh giá ----------------
test_pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
mae = mean_absolute_error(y_test_real, test_pred)
r2 = r2_score(y_test_real, test_pred)

st.subheader("Kết quả đánh giá")
c1, c2, c3 = st.columns(3)
c1.metric("RMSE", f"{rmse:,.2f}")
c2.metric("MAE", f"{mae:,.2f}")
c3.metric("R²", f"{r2:.4f}")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Loss: Train vs Validation**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.legend()
    st.pyplot(fig1)

with col_b:
    st.markdown("**Giá thật vs Dự báo (tập test)**")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(y_test_real, label="Giá thật")
    ax2.plot(test_pred, label="Dự báo")
    ax2.legend()
    st.pyplot(fig2)

# ---------------- Dự báo tương lai ----------------
last_window = np.vstack((train_scaled, test_scaled))[-window_size:]
last_window = last_window.reshape(1, window_size, 1)

future = []
for _ in range(n_future):
    nxt = model.predict(last_window, verbose=0)
    future.append(nxt[0, 0])
    last_window = np.append(last_window[:, 1:, :], nxt.reshape(1, 1, 1), axis=1)

future = scaler.inverse_transform(np.array(future).reshape(-1, 1))
future_dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=n_future)
future_df = pd.DataFrame({"Ngày": future_dates.date, "Giá dự báo": future.flatten().round(2)})

st.subheader(f"Dự báo {n_future} ngày tới")
col_c, col_d = st.columns([1, 2])
col_c.dataframe(future_df, hide_index=True)

with col_d:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(y_test_real, label="Giá thật")
    ax3.plot(test_pred, label="Dự báo test")
    ax3.plot(range(len(y_test_real), len(y_test_real) + n_future),
             future, marker="o", label=f"{n_future} ngày tới")
    ax3.legend()
    st.pyplot(fig3)

st.caption("⚠️ Chỉ dùng cho mục đích học tập, không phải khuyến nghị đầu tư.")
