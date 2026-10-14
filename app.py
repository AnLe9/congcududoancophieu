import random
import traceback
from datetime import date, timedelta

import altair as alt
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
UNITS = {
    "Đồng (đầy đủ số, ví dụ 62,500 ₫)": (1.0, "₫", 0),
    "Nghìn đồng (ví dụ 62.50 nghìn ₫)": (0.001, "nghìn ₫", 2),
    "Nhân 1.000 (nếu giá gốc đang tính bằng nghìn đồng)": (1000.0, "₫", 0),
    "Số gốc, không ghi đơn vị (cổ phiếu nước ngoài)": (1.0, "", 2),
}
MODEL_COLORS = {"LSTM": "#ff7f0e", "GRU": "#2ca02c"}
MODEL_LIGHT = {"LSTM": "#ffbb78", "GRU": "#98df8a"}

# =============== CẤU HÌNH ĐƠN GIẢN ===============
st.sidebar.header("⚙️ Cài đặt")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu (ví dụ: VNM.VN)",
    "VNM.VN"
).strip().upper()

start_date = st.sidebar.date_input("Dữ liệu từ ngày", date(2018, 1, 1))
end_date = st.sidebar.date_input(
    "Đến ngày", date.today(), max_value=date.today()
)

unit_name = st.sidebar.selectbox(
    "Đơn vị hiển thị giá",
    list(UNITS.keys()),
    index=0 if ticker.endswith(".VN") else 3,
    help="Nếu giá hiển thị quá nhỏ hoặc quá lớn, hãy đổi đơn vị."
)
K, UNIT, DEC = UNITS[unit_name]
UNIT_TXT = f" ({UNIT})" if UNIT else ""
FMT = f",.{DEC}f"

n_future = st.sidebar.slider(
    "Dự báo bao nhiêu ngày tiếp theo?",
    min_value=1, max_value=30, value=10
)

# Cấu hình AI tự động để người dùng không cần hiểu thuật ngữ kỹ thuật.
model_choices = ["LSTM"]
extra_feats = ["Volume", "MA20", "RSI", "MACD"]
window_size = 60
epochs = 20
seed = 42

if start_date >= end_date:
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()


def money(x):
    """Định dạng giá theo đơn vị người dùng chọn."""
    return f"{x * K:,.{DEC}f} {UNIT}".strip()


@st.cache_data(show_spinner="Đang tải dữ liệu giá...", ttl=3600)
def load_data(tk, start, end):
    df = yf.download(
        tk, start=str(start), end=str(end + timedelta(days=1)),
        progress=False, auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df[cols].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


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


def train_and_forecast(df_raw, features, window, epochs, n_future, model_type, seed):
    set_seed(int(seed))
    d = add_indicators(df_raw).dropna()
    if len(d) < window + 100:
        raise ValueError(
            "Dữ liệu quá ít. Hãy chọn khoảng thời gian dài hơn."
        )

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
    X_test = X[n_train:]

    val_size = max(int(0.1 * len(X_train)), 1)
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    Layer = LSTM if model_type == "LSTM" else GRU
    model = Sequential([
        Input(shape=(window, len(features))),
        Layer(64, return_sequences=True),
        Dropout(0.2),
        Layer(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=64, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )],
    )

    # Kiểm tra trên phần dữ liệu AI chưa từng học.
    pred = sc_close.inverse_transform(
        model.predict(X_test, verbose=0)
    ).flatten()
    real = close[split:, 0]
    prev = close[split - 1:len(d) - 1, 0]
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    rmse_naive = float(np.sqrt(mean_squared_error(real, prev)))
    mae = float(mean_absolute_error(real, pred))
    mape = float(np.mean(np.abs((real - pred) / np.maximum(np.abs(real), 1e-10))) * 100)
    r2 = float(r2_score(real, pred))
    dir_acc = float(np.mean(np.sign(real - prev) == np.sign(pred - prev)) * 100)
    sigma = float(np.std(real - pred))

    # Dự báo từng ngày. Các chỉ báo tương lai được tính lại từ giá mô hình dự báo.
    ext = d[["Open", "High", "Low", "Close", "Volume"]].copy()
    future_dates = pd.bdate_range(
        d.index[-1] + pd.Timedelta(days=1), periods=n_future
    )
    future = []
    for dt in future_dates:
        tmp = add_indicators(ext)
        # Bảo đảm có đủ dữ liệu và không có giá trị thiếu ở cửa sổ cuối.
        tmp_window = tmp[features].tail(window)
        if len(tmp_window) < window or tmp_window.isna().any().any():
            raise ValueError("Không đủ dữ liệu để tạo dự báo. Hãy chọn khoảng ngày dài hơn.")
        win = sc_all.transform(tmp_window.values)
        p_s = model.predict(
            win.reshape(1, window, len(features)), verbose=0
        )[0, 0]
        p = float(sc_close.inverse_transform([[p_s]])[0, 0])
        p = max(p, 0.0)
        future.append(p)
        last_volume = float(ext["Volume"].iloc[-1])
        ext.loc[dt, ["Open", "High", "Low", "Close", "Volume"]] = [
            p, p, p, p, last_volume
        ]

    future = np.array(future)
    steps = np.arange(1, n_future + 1)
    band = 1.96 * sigma * np.sqrt(steps)

    names = ", ".join(
        ["Giá đóng cửa"] + [FEAT_NAMES[f] for f in features if f in FEAT_NAMES]
    )
    return {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "test_dates": d.index[split:],
        "pred": pred,
        "hist_close": d["Close"],
        "future_dates": future_dates,
        "future": future,
        "lower": np.maximum(future - band, 0),
        "upper": future + band,
        "metrics": dict(
            rmse=rmse, rmse_naive=rmse_naive, mae=mae,
            mape=mape, r2=r2, dir_acc=dir_acc
        ),
        "info": f"AI sử dụng {len(d)} ngày dữ liệu lịch sử",
        "last_date": d.index[-1],
    }


# =============== BIỂU ĐỒ ===============
XD = alt.X("date:T", title=None)


def yq(field):
    return alt.Y(
        f"{field}:Q",
        scale=alt.Scale(zero=False),
        title=f"Giá{UNIT_TXT}",
        axis=alt.Axis(format=FMT)
    )


def show(chart):
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart)


def plot_df(df):
    p = df.copy()
    price_cols = [
        c for c in [
            "Open", "High", "Low", "Close", "MA20", "MA50",
            "BB_up", "BB_low", "MACD", "Signal", "Hist"
        ] if c in p.columns
    ]
    p[price_cols] = p[price_cols] * K
    p.index.name = "date"
    p = p.reset_index()
    p.columns = [c.lower() for c in p.columns]
    return p


def candle_chart(p, show_ma, show_bb):
    n = len(p)
    bar_size = int(max(2, min(10, 700 / max(n, 1) * 0.7)))
    up = alt.condition(
        "datum.open <= datum.close",
        alt.value("#26a69a"),
        alt.value("#ef5350")
    )
    tip = [
        alt.Tooltip("date:T", title="Ngày"),
        alt.Tooltip("open:Q", title="Mở cửa", format=FMT),
        alt.Tooltip("high:Q", title="Cao nhất", format=FMT),
        alt.Tooltip("low:Q", title="Thấp nhất", format=FMT),
        alt.Tooltip("close:Q", title="Đóng cửa", format=FMT),
        alt.Tooltip("volume:Q", title="Khối lượng", format=",.0f"),
    ]
    base = alt.Chart(p)
    layers = []

    if show_bb:
        layers.append(
            base.mark_area(opacity=0.12, color="gray").encode(
                x=XD, y=yq("bb_low"), y2="bb_up:Q"
            )
        )

    layers.append(
        base.mark_rule().encode(
            x=XD, y=yq("low"), y2="high:Q", color=up, tooltip=tip
        )
    )
    layers.append(
        base.mark_bar(size=bar_size).encode(
            x=XD, y=yq("open"), y2="close:Q", color=up, tooltip=tip
        )
    )

    if show_ma:
        ma = (
            base.transform_fold(["ma20", "ma50"], as_=["k", "v"])
            .transform_calculate(
                name="datum.k == 'ma20' ? 'Trung bình 20 ngày' : 'Trung bình 50 ngày'"
            )
            .mark_line(strokeWidth=1.5)
            .encode(
                x=XD,
                y=yq("v"),
                color=alt.Color(
                    "name:N", title=None,
                    scale=alt.Scale(
                        domain=["Trung bình 20 ngày", "Trung bình 50 ngày"],
                        range=["#ff9800", "#2196f3"]
                    )
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Ngày"),
                    alt.Tooltip("name:N", title="Đường"),
                    alt.Tooltip("v:Q", title="Giá trị", format=FMT)
                ]
            )
        )
        layers.append(ma)

    return alt.layer(*layers).properties(
        height=380, width="container"
    ).interactive()


def volume_chart(p):
    up = alt.condition(
        "datum.open <= datum.close",
        alt.value("#26a69a"),
        alt.value("#ef5350")
    )
    return (
        alt.Chart(p).mark_bar()
        .encode(
            x=XD,
            y=alt.Y(
                "volume:Q", title="Khối lượng giao dịch",
                axis=alt.Axis(format=",.0f")
            ),
            color=up,
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("volume:Q", title="Khối lượng", format=",.0f")
            ]
        )
        .properties(height=140, width="container")
    )


def rsi_chart(p):
    line = (
        alt.Chart(p).mark_line()
        .encode(
            x=XD,
            y=alt.Y("rsi:Q", title="Sức mạnh giá (RSI)",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("rsi:Q", title="RSI", format=".1f")
            ]
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({
            "y": [30, 70], "c": ["#2e7d32", "#c62828"]
        }))
        .mark_rule(strokeDash=[5, 5])
        .encode(y="y:Q", color=alt.Color("c:N", scale=None, legend=None))
    )
    return alt.layer(line, rules).properties(
        height=240, width="container"
    ).interactive()


def macd_chart(p):
    hist = (
        alt.Chart(p).mark_bar(opacity=0.5)
        .encode(
            x=XD,
            y=alt.Y("hist:Q", title=f"Đà tăng/giảm (MACD){UNIT_TXT}",
                    axis=alt.Axis(format=FMT)),
            color=alt.condition(
                "datum.hist >= 0",
                alt.value("#26a69a"),
                alt.value("#ef5350")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("hist:Q", title="Chênh lệch", format=FMT)
            ]
        )
    )
    lines = (
        alt.Chart(p)
        .transform_fold(["macd", "signal"], as_=["k", "v"])
        .transform_calculate(
            name="datum.k == 'macd' ? 'MACD' : 'Đường tín hiệu'"
        )
        .mark_line(strokeWidth=1.5)
        .encode(
            x=XD,
            y=alt.Y("v:Q", title=f"Đà tăng/giảm (MACD){UNIT_TXT}",
                    axis=alt.Axis(format=FMT)),
            color=alt.Color(
                "name:N", title=None,
                scale=alt.Scale(
                    domain=["MACD", "Đường tín hiệu"],
                    range=["#2196f3", "#ff9800"]
                )
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("name:N", title="Đường"),
                alt.Tooltip("v:Q", title="Giá trị", format=FMT)
            ]
        )
    )
    return alt.layer(hist, lines).properties(
        height=260, width="container"
    ).interactive()


def forecast_chart(results):
    names = list(results.keys())
    first = results[names[0]]

    hist = first["hist_close"].tail(250)
    actual = pd.DataFrame({
        "date": hist.index,
        "price": hist.values * K,
    })

    backtest_rows, future_rows, band_rows = [], [], []

    for name, r in results.items():
        backtest_rows.append(pd.DataFrame({
            "date": r["test_dates"],
            "price": r["pred"] * K,
            "model": name,
        }))

        # Điểm đầu tiên là giá thực tế gần nhất, sau đó là các ngày tương lai.
        future_dates = pd.DatetimeIndex(
            [r["last_date"]]
        ).append(pd.DatetimeIndex(r["future_dates"]))
        future_prices = np.concatenate([
            [float(r["hist_close"].iloc[-1])],
            r["future"]
        ]) * K

        future_rows.append(pd.DataFrame({
            "date": future_dates,
            "price": future_prices,
            "model": name,
        }))

        band_rows.append(pd.DataFrame({
            "date": r["future_dates"],
            "lower": r["lower"] * K,
            "upper": r["upper"] * K,
            "model": name,
        }))

    backtest = pd.concat(backtest_rows, ignore_index=True)
    future = pd.concat(future_rows, ignore_index=True)
    bands = pd.concat(band_rows, ignore_index=True)

    actual_chart = (
        alt.Chart(actual).mark_line(color="#1976d2", strokeWidth=2)
        .encode(
            x=XD, y=yq("price"),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("price:Q", title="Giá thực tế", format=FMT)
            ]
        )
    )

    backtest_chart = (
        alt.Chart(backtest).mark_line(strokeWidth=1.5, opacity=0.6)
        .encode(
            x=XD, y=yq("price"),
            color=alt.Color(
                "model:N", title="AI kiểm tra dữ liệu cũ",
                scale=alt.Scale(
                    domain=names, range=[MODEL_LIGHT[n] for n in names]
                )
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("model:N", title="Mô hình"),
                alt.Tooltip("price:Q", title="AI dự đoán lại", format=FMT)
            ]
        )
    )

    band_chart = (
        alt.Chart(bands).mark_area(opacity=0.18)
        .encode(
            x=XD, y=yq("lower"), y2="upper:Q",
            color=alt.Color(
                "model:N", title=None,
                scale=alt.Scale(
                    domain=names, range=[MODEL_COLORS[n] for n in names]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("model:N", title="Mô hình"),
                alt.Tooltip("lower:Q", title="Khoảng thấp", format=FMT),
                alt.Tooltip("upper:Q", title="Khoảng cao", format=FMT)
            ]
        )
    )

    # Đường đậm thể hiện dự báo tương lai, nối trực tiếp từ giá gần nhất.
    future_chart = (
        alt.Chart(future).mark_line(strokeWidth=3)
        .encode(
            x=XD, y=yq("price"),
            color=alt.Color(
                "model:N", title="Đường dự báo tương lai",
                scale=alt.Scale(
                    domain=names, range=[MODEL_COLORS[n] for n in names]
                )
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("model:N", title="Mô hình"),
                alt.Tooltip("price:Q", title="Giá dự báo", format=FMT)
            ]
        )
    )

    future_points_data = future[future["date"] > first["last_date"]]
    future_points = (
        alt.Chart(future_points_data).mark_point(filled=True, size=65)
        .encode(
            x=XD, y=yq("price"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(
                    domain=names, range=[MODEL_COLORS[n] for n in names]
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("model:N", title="Mô hình"),
                alt.Tooltip("price:Q", title="Giá dự báo", format=FMT)
            ]
        )
    )

    boundary = (
        alt.Chart(pd.DataFrame({"date": [first["last_date"]]}))
        .mark_rule(strokeDash=[5, 5], color="gray")
        .encode(x=XD)
    )

    return alt.layer(
        band_chart, actual_chart, backtest_chart,
        future_chart, future_points, boundary
    ).properties(height=430, width="container").interactive()


def loss_chart(results):
    rows = []
    for name, r in results.items():
        n = len(r["loss"])
        rows.append(pd.DataFrame({
            "lan": range(1, n + 1),
            "v": r["loss"],
            "k": f"{name}: AI học"
        }))
        rows.append(pd.DataFrame({
            "lan": range(1, n + 1),
            "v": r["val_loss"],
            "k": f"{name}: AI tự kiểm tra"
        }))
    df = pd.concat(rows, ignore_index=True)
    return (
        alt.Chart(df).mark_line()
        .encode(
            x=alt.X("lan:Q", title="Lần học"),
            y=alt.Y("v:Q", title="Sai số"),
            color=alt.Color("k:N", title=None),
            tooltip=[
                alt.Tooltip("k:N", title="Loại"),
                alt.Tooltip("lan:Q", title="Lần học"),
                alt.Tooltip("v:Q", title="Sai số", format=".5f")
            ]
        )
        .properties(height=250, width="container")
    )


def compare_chart(norm):
    n = norm.copy()
    n.index.name = "date"
    long = n.reset_index().melt(
        id_vars="date", var_name="ma", value_name="v"
    )
    return (
        alt.Chart(long).mark_line(strokeWidth=2)
        .encode(
            x=XD,
            y=alt.Y(
                "v:Q", scale=alt.Scale(zero=False),
                title="Điểm (ngày đầu = 100)"
            ),
            color=alt.Color("ma:N", title="Mã cổ phiếu"),
            tooltip=[
                alt.Tooltip("date:T", title="Ngày"),
                alt.Tooltip("ma:N", title="Mã"),
                alt.Tooltip("v:Q", title="Điểm", format=".1f")
            ]
        )
        .properties(height=380, width="container").interactive()
    )


def corr_chart(corr):
    c = corr.copy()
    c.index.name = "a"
    c.columns.name = "b"
    long = c.reset_index().melt(
        id_vars="a", var_name="b", value_name="v"
    )
    base = alt.Chart(long).encode(
        x=alt.X("b:N", title=None),
        y=alt.Y("a:N", title=None)
    )
    rect = base.mark_rect().encode(
        color=alt.Color(
            "v:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
            title="Mức tương quan"
        ),
        tooltip=[
            alt.Tooltip("a:N", title="Mã 1"),
            alt.Tooltip("b:N", title="Mã 2"),
            alt.Tooltip("v:Q", title="Tương quan", format=".2f")
        ]
    )
    text = base.mark_text().encode(
        text=alt.Text("v:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.v) > 0.6", alt.value("white"), alt.value("black")
        )
    )
    return (rect + text).properties(height=280, width="container")


# =============== TẢI DỮ LIỆU ===============
df_raw = load_data(ticker, start_date, end_date)

if df_raw.empty or len(df_raw) < 300:
    st.error(
        "Không tải được đủ dữ liệu (cần khoảng 300 ngày giao dịch trở lên). "
        "Hãy kiểm tra mã cổ phiếu hoặc chọn khoảng thời gian dài hơn."
    )
    st.stop()

df_ind = add_indicators(df_raw)

st.markdown(
    f"**Mã cổ phiếu:** {ticker}  |  "
    f"**Dữ liệu:** {df_raw.index[0].date()} → {df_raw.index[-1].date()}"
)

if st.button("🚀 Dự báo giá cổ phiếu", type="primary"):
    try:
        features = ["Close"] + extra_feats
        results = {}

        with st.spinner("AI đang học dữ liệu và tính toán dự báo..."):
            for model_name in model_choices:
                results[model_name] = train_and_forecast(
                    df_raw, features, window_size, epochs,
                    n_future, model_name, seed
                )

        st.session_state["res"] = results
        st.session_state["res_ticker"] = ticker
        st.session_state["res_n_future"] = n_future
        st.success("Đã hoàn tất! Mở tab 'Dự báo giá' để xem kết quả.")

    except Exception as e:
        st.error(f"Chưa thể dự báo: {e}")
        st.exception(e)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Giá cổ phiếu",
    "📐 Chỉ báo tham khảo",
    "🔮 Dự báo giá",
    "🔀 So sánh cổ phiếu"
])

# =============== TAB 1: GIÁ CỔ PHIẾU ===============
with tab1:
    last = df_raw["Close"].iloc[-1]
    prev_close = df_raw["Close"].iloc[-2]
    last_year = df_raw.tail(252)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Giá đóng cửa gần nhất", money(last),
        f"{(last / prev_close - 1) * 100:.2f}%"
    )
    c2.metric("Giá cao nhất 1 năm", money(last_year["High"].max()))
    c3.metric("Giá thấp nhất 1 năm", money(last_year["Low"].min()))
    c4.metric(
        "Khối lượng giao dịch TB 20 ngày",
        f"{df_raw['Volume'].tail(20).mean():,.0f} cổ phiếu"
    )

    show_n = st.slider(
        "Số ngày hiển thị", 60, min(len(df_ind), 500), 120, 10
    )
    show_ma = st.checkbox("Hiện đường giá trung bình", True)
    show_bb = st.checkbox("Hiện vùng dao động giá", False)
    p = plot_df(df_ind.tail(show_n))

    st.caption(
        "Nến xanh: giá đóng cửa cao hơn hoặc bằng giá mở cửa. "
        "Nến đỏ: giá đóng cửa thấp hơn giá mở cửa. "
        "Chạm vào biểu đồ để xem chi tiết."
    )
    show(candle_chart(p, show_ma, show_bb))
    show(volume_chart(p))

    with st.expander("Xem và tải dữ liệu"):
        st.dataframe(df_raw.tail(30).round(2))
        st.download_button(
            "⬇️ Tải dữ liệu CSV",
            df_raw.to_csv().encode("utf-8-sig"),
            f"{ticker}_data.csv",
            "text/csv"
        )

# =============== TAB 2: CHỈ BÁO THAM KHẢO ===============
with tab2:
    v = df_ind.dropna().tail(250)
    rsi_now = float(v["RSI"].iloc[-1])
    macd_now = float(v["MACD"].iloc[-1])
    sig_now = float(v["Signal"].iloc[-1])

    st.markdown("### Nhận định tham khảo")

    if rsi_now > 70:
        st.warning(
            f"Chỉ số sức mạnh giá là {rsi_now:.1f}. "
            "Giá đã tăng mạnh trong thời gian gần đây và có thể điều chỉnh."
        )
    elif rsi_now < 30:
        st.info(
            f"Chỉ số sức mạnh giá là {rsi_now:.1f}. "
            "Giá đã giảm mạnh trong thời gian gần đây; khả năng hồi phục không được đảm bảo."
        )
    else:
        st.info(
            f"Chỉ số sức mạnh giá là {rsi_now:.1f}, "
            "nằm trong vùng thường gặp từ 30 đến 70."
        )

    if macd_now > sig_now:
        st.write(
            "Chỉ báo đà giá đang nằm trên đường tín hiệu, "
            "cho thấy đà tăng gần đây đang mạnh hơn."
        )
    else:
        st.write(
            "Chỉ báo đà giá đang nằm dưới đường tín hiệu, "
            "cho thấy đà giảm gần đây đang mạnh hơn."
        )

    pv = plot_df(v)
    st.markdown("**Sức mạnh giá**")
    show(rsi_chart(pv))
    st.markdown("**Đà tăng và giảm của giá**")
    show(macd_chart(pv))

# =============== TAB 3: DỰ BÁO GIÁ ===============
with tab3:
    res = st.session_state.get("res")

    if not res:
        st.info(
            "Nhập mã cổ phiếu ở thanh bên, chọn số ngày muốn dự báo, "
            "rồi bấm '🚀 Dự báo giá cổ phiếu'. Kết quả sẽ xuất hiện ở đây."
        )
    elif st.session_state.get("res_ticker") != ticker:
        st.info(
            "Mã cổ phiếu đã thay đổi. Hãy bấm '🚀 Dự báo giá cổ phiếu' "
            "để tính kết quả cho mã mới."
        )
    else:
        names = list(res.keys())
        first = res[names[0]]
        last_close = float(first["hist_close"].iloc[-1])
        end_day = first["future_dates"][-1].strftime("%d/%m/%Y")

        st.caption(
            f"Kết quả cho {st.session_state['res_ticker']} — "
            f"{first['info']}."
        )

        st.markdown(f"### 🔮 Giá dự kiến đến ngày {end_day}")

        for name in names:
            r = res[name]
            predicted = float(r["future"][-1])
            change = (predicted / last_close - 1) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Giá đóng cửa gần nhất", money(last_close))
            c2.metric(
                f"Giá dự kiến sau {len(r['future'])} ngày",
                money(predicted),
                f"{change:+.2f}% so với giá gần nhất"
            )
            c3.metric(
                "Khoảng giá tham khảo",
                f"{money(float(r['lower'][-1]))} – "
                f"{money(float(r['upper'][-1]))}"
            )

        st.markdown("### 📈 Biểu đồ giá thực tế và dự báo")
        st.caption(
            "Đường xanh dương thể hiện giá thực tế. Đường màu cam thể hiện "
            "giá AI dự báo trong tương lai; các chấm là từng ngày dự báo. "
            "Vùng tô màu là khoảng giá tham khảo, không phải mức giá chắc chắn. "
            "Đường nét đứt đánh dấu nơi bắt đầu dự báo."
        )
        show(forecast_chart(res))

        st.markdown("### 📅 Giá dự kiến từng ngày")
        disp = pd.DataFrame({
            "Ngày": [
                d.strftime("%d/%m/%Y") for d in first["future_dates"]
            ]
        })
        numeric = pd.DataFrame({
            "Ngày": first["future_dates"].date
        })

        for name in names:
            r = res[name]
            base_price = float(r["hist_close"].iloc[-1])
            predicted_col = f"{name} - Giá dự kiến"
            low_col = f"{name} - Mức thấp tham khảo"
            high_col = f"{name} - Mức cao tham khảo"
            change_col = f"{name} - Tăng/giảm so với hiện tại"

            numeric[predicted_col] = np.round(r["future"] * K, DEC)
            numeric[low_col] = np.round(r["lower"] * K, DEC)
            numeric[high_col] = np.round(r["upper"] * K, DEC)
            numeric[change_col] = np.round(
                (r["future"] / base_price - 1) * 100, 2
            )

            disp[predicted_col] = [
                f"{x:,.{DEC}f}" for x in r["future"] * K
            ]
            disp[low_col] = [
                f"{x:,.{DEC}f}" for x in r["lower"] * K
            ]
            disp[high_col] = [
                f"{x:,.{DEC}f}" for x in r["upper"] * K
            ]
            disp[change_col] = [
                f"{x:+.2f}%"
                for x in (r["future"] / base_price - 1) * 100
            ]

        st.dataframe(disp, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Tải bảng dự báo",
            numeric.to_csv(index=False).encode("utf-8-sig"),
            f"{ticker}_du_bao.csv",
            "text/csv"
        )

        with st.expander("Độ chính xác khi kiểm tra dữ liệu cũ"):
            st.caption(
                "Các chỉ số dưới đây đo hiệu quả trên phần dữ liệu được giữ lại "
                "để kiểm tra, không đảm bảo độ chính xác trong tương lai."
            )
            acc_rows = []
            for name in names:
                m = res[name]["metrics"]
                acc_rows.append({
                    "Mô hình": name,
                    "Sai số giá (RMSE)": f"{m['rmse'] * K:,.{DEC}f}",
                    "Sai lệch trung bình (MAE)": f"{m['mae'] * K:,.{DEC}f}",
                    "Sai lệch trung bình (%)": f"{m['mape']:.2f}%",
                    "Mức độ khớp (R²)": f"{m['r2']:.3f}",
                    "Đoán đúng hướng tăng/giảm": f"{m['dir_acc']:.1f}%",
                    "So với cách lấy giá hôm nay làm giá ngày mai":
                        "Tốt hơn" if m["rmse"] < m["rmse_naive"]
                        else "Chưa tốt hơn",
                })
            st.dataframe(
                pd.DataFrame(acc_rows),
                hide_index=True,
                use_container_width=True
            )

        with st.expander("Quá trình AI học"):
            st.caption(
                "Biểu đồ này dành cho người muốn xem chi tiết cách mô hình học."
            )
            show(loss_chart(res))

        st.warning(
            "Dự báo chỉ mang tính tham khảo và phục vụ học tập, "
            "không phải lời khuyên mua hoặc bán cổ phiếu. "
            "Khoảng giá là ước lượng đơn giản, chưa phải khoảng tin cậy "
            "được hiệu chuẩn thống kê."
        )

# =============== TAB 4: SO SÁNH CỔ PHIẾU ===============
with tab4:
    tickers_text = st.text_input(
        "Nhập ít nhất 2 mã, cách nhau bằng dấu phẩy",
        "VNM.VN, FPT.VN, HPG.VN"
    )
    cmp_start = st.date_input(
        "So sánh từ ngày",
        max(start_date, end_date - timedelta(days=365 * 2)),
        key="cmp_start"
    )
    tickers = [
        t.strip().upper()
        for t in tickers_text.split(",")
        if t.strip()
    ]

    closes = {}
    for t in tickers:
        try:
            dft = load_data(t, cmp_start, end_date)
            if not dft.empty:
                closes[t] = dft["Close"]
            else:
                st.warning(f"Không tải được dữ liệu cho mã {t}.")
        except Exception:
            st.warning(f"Không tải được dữ liệu cho mã {t}.")

    if len(closes) >= 2:
        cdf = pd.DataFrame(closes).dropna()

        if not cdf.empty:
            norm = cdf / cdf.iloc[0] * 100
            st.markdown(
                "**Mã nào tăng tốt hơn?** "
                "Mỗi mã được quy về 100 điểm tại ngày đầu so sánh."
            )
            show(compare_chart(norm))

            summary = pd.DataFrame({
                "Lãi/lỗ cả giai đoạn": [
                    f"{x:+.2f}%"
                    for x in (cdf.iloc[-1] / cdf.iloc[0] - 1) * 100
                ],
                "Mức dao động giá mỗi năm": [
                    f"{x:.2f}%"
                    for x in cdf.pct_change().std() * np.sqrt(252) * 100
                ],
            }, index=cdf.columns)
            st.dataframe(summary, use_container_width=True)
            st.caption(
                "Mức dao động càng cao, giá cổ phiếu biến động càng mạnh; "
                "đây là thông tin tham khảo về rủi ro."
            )

            st.markdown("**Các mã có xu hướng tăng giảm cùng nhau không?**")
            show(corr_chart(cdf.pct_change().dropna().corr()))
        else:
            st.info("Không có đủ dữ liệu trùng ngày để so sánh.")
    else:
        st.info("Nhập ít nhất 2 mã cổ phiếu hợp lệ để so sánh.")
