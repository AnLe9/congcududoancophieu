import random
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from keras.callbacks import EarlyStopping
from keras.layers import GRU, LSTM, Dense, Dropout, Input
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Phân tích & dự báo cổ phiếu", layout="wide")
st.title("📈 Phân tích & dự báo cổ phiếu bằng AI")

# =============== SIDEBAR ===============
st.sidebar.header("⚙️ Cấu hình")
ticker = st.sidebar.text_input("Mã cổ phiếu (Yahoo Finance)", "VNM.VN").strip().upper()
start_date = st.sidebar.date_input("Từ ngày", date(2018, 1, 1))

st.sidebar.subheader("Mô hình")
model_type = st.sidebar.selectbox("Loại mô hình", ["LSTM", "GRU"])
extra_feats = st.sidebar.multiselect(
    "Đặc trưng thêm (ngoài giá đóng cửa)",
    ["Volume", "MA20", "RSI", "MACD"],
    default=["Volume", "MA20", "RSI"],
)
window_size = st.sidebar.slider("Số ngày nhìn lại (window)", 20, 120, 60, 10)
epochs = st.sidebar.slider("Epochs", 5, 100, 30)
n_future = st.sidebar.slider("Số ngày dự báo", 1, 30, 10)
seed = st.sidebar.number_input("Seed", value=42, step=1)


# =============== HÀM HỖ TRỢ ===============
@st.cache_data(show_spinner="Đang tải dữ liệu...", ttl=3600)
def load_data(tk, start):
    df = yf.download(
        tk, start=str(start), end=str(date.today() + timedelta(days=1)),
        progress=False, auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols].dropna()


def add_indicators(df):
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    std20 = d["Close"].rolling(20).std()
    d["BB_up"] = d["MA20"] + 2 * std20
    d["BB_low"] = d["MA20"] - 2 * std20
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-10)
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["Hist"] = d["MACD"] - d["Signal"]
    return d


def set_seed(s):
    np.random.seed(s)
    tf.random.set_seed(s)
    random.seed(s)


def draw_candles(ax, v):
    up = v["Close"] >= v["Open"]
    colors = np.where(up, "#26a69a", "#ef5350")
    ax.vlines(v.index, v["Low"], v["High"], color=colors, linewidth=0.8)
    height = (v["Close"] - v["Open"]).abs().replace(0, 1e-6)
    bottom = v[["Open", "Close"]].min(axis=1)
    ax.bar(v.index, height, bottom=bottom, width=0.7, color=colors)


def train_and_forecast(df_raw, features, window, epochs, n_future, model_type, seed):
    set_seed(int(seed))
    d = add_indicators(df_raw).dropna()
    feat = d[features].values
    close = d[["Close"]].values
    split = int(0.8 * len(d))

    sc_all = MinMaxScaler().fit(feat[:split])
    sc_close = MinMaxScaler().fit(close[:split])
    feat_s = sc_all.transform(feat)
    close_s = sc_close.transform(close)

    X, y = [], []
    for i in range(window, len(d)):
        X.append(feat_s[i - window:i])
        y.append(close_s[i, 0])
    X, y = np.array(X), np.array(y)

    n_train = split - window
    X_train, y_train = X[:n_train], y[:n_train]
