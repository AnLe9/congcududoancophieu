import random
import traceback
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
st.title("📈 Phân tích & dự báo giá cổ phiếu")

FEAT_NAMES = {
    "Volume": "Khối lượng giao dịch",
    "MA20": "Đường trung bình 20 ngày",
    "RSI": "RSI (sức mạnh giá)",
    "MACD": "MACD (đà tăng/giảm)",
}

# =============== THANH BÊN: CẤU HÌNH ===============
st.sidebar.header("⚙️ Cấu hình")
ticker = st.sidebar.text_input("Mã cổ phiếu (ví dụ: VNM.VN)", "VNM.VN").strip().upper()
start_date = st.sidebar.date_input("Từ ngày", date(2018, 1, 1))
end_date = st.sidebar.date_input("Đến ngày", date.today(), max_value=date.today())

st.sidebar.subheader("Cách dự báo")
model_type = st.sidebar.selectbox(
    "Kiểu mô hình AI", ["LSTM", "GRU"],
    help="Hai kiểu AI chuyên đọc dữ liệu theo thời gian. GRU nhẹ và chạy nhanh hơn LSTM một chút.")
extra_feats = st.sidebar.multiselect(
    "Thông tin thêm cho AI học (ngoài giá đóng cửa)",
    list(FEAT_NAMES.keys()), default=["Volume", "MA20", "RSI"],
    format_func=lambda x: FEAT_NAMES[x],
    help="Càng nhiều thông tin, AI càng có nhiều dữ liệu để học, nhưng chưa chắc đoán đúng hơn.")
window_size = st.sidebar.slider(
    "Số ngày gần nhất AI dùng để đoán", 20, 120, 60, 10,
    help="Ví dụ 60 nghĩa là AI nhìn lại 60 ngày giao dịch gần nhất để đoán ngày kế tiếp.")
epochs = st.sidebar.slider(
    "Số lần học", 5, 100, 20,
    help="Số lần AI học lại dữ liệu. Nhiều hơn thì có thể chính xác hơn nhưng chạy lâu hơn.")
n_future = st.sidebar.slider("Số ngày muốn dự báo tiếp theo", 1, 30, 10)
seed = st.sidebar.number_input(
    "Số cố định kết quả", value=42, step=1,
    help="Giữ nguyên số này thì mỗi lần chạy sẽ ra kết quả giống nhau.")

if start_date >= end_date:
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()


# =============== HÀM HỖ TRỢ ===============
@st.cache_data(show_spinner="Đang tải dữ liệu giá...", ttl=3600)
def load_data(tk, start, end):
    df = yf.download(
        tk, start=str(start), end=str(end + timedelta(days=1)),
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
    d["MAC
