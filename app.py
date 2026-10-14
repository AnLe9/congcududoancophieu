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

# =============== THANH BÊN: CẤU HÌNH ===============
st.sidebar.header("⚙️ Cấu hình")
ticker = st.sidebar.text_input("Mã cổ phiếu (ví dụ: VNM.VN)", "VNM.VN").strip().upper()
start_date = st.sidebar.date_input("Từ ngày", date(2018, 1, 1))
end_date = st.sidebar.date_input("Đến ngày", date.today(), max_value=date.today())
unit_name = st.sidebar.selectbox(
    "Đơn vị hiển thị giá", list(UNITS.keys()),
    index=0 if ticker.endswith(".VN") else 3,
    help="Nếu thấy giá hiển thị quá nhỏ hoặc quá lớn so với thực tế, hãy đổi đơn vị ở đây.")
K, UNIT, DEC = UNITS[unit_name]
UNIT_TXT = f" ({UNIT})" if UNIT else ""
FMT = f",.{DEC}f"

st.sidebar.subheader("Cách dự báo")
model_choices = st.sidebar.multiselect(
    "Kiểu AI dự báo", ["LSTM", "GRU"], default=["LSTM"],
    help="Hai kiểu AI chuyên đọc dữ liệu theo thời gian. Chọn cả hai để so sánh kết quả (chạy lâu gấp đôi).")
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


def money(x):
    """Định dạng 1 mức giá: đầy đủ số + đơn vị."""
    return f"{x * K:,.{DEC}f} {UNIT}".strip()


# =============== HÀM DỮ LIỆU & AI ===============
@st.cache_data(show_spinner="Đang tải dữ liệu giá...", ttl=3600)
def load_data(tk, start, end):
    df = yf.download(
        tk, start=str(start), end=str(end + timedelta(days=1)),
        progress=False, auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
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
        raise ValueError("Dữ liệu quá ít so với 'số ngày AI dùng để đoán'. "
                         "Hãy chọn khoảng thời gian dài hơn hoặc giảm số ngày này.")
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
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    # ----- Chấm điểm trên phần dữ liệu AI chưa từng thấy -----
    pred = sc_close.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    real = close[split:, 0]
    prev = close[split - 1:len(d) - 1, 0]
    rmse = float(np.sqrt(mean_squared_error(real, pred)))
    rmse_naive = float(np.sqrt(mean_squared_error(real, prev)))
    mae = float(mean_absolute_error(real, pred))
    mape = float(np.mean(np.abs((real - pred) / real)) * 100)
    r2 = float(r2_score(real, pred))
    dir_acc = float(np.mean(np.sign(real - prev) == np.sign(pred - prev)) * 100)
    sigma = float(np.std(real - pred))

    # ----- Dự báo các ngày tiếp theo -----
    ext = d[["Open", "High", "Low", "Close", "Volume"]].copy()
    future_dates = pd.bdate_range(d.index[-1] + pd.Timedelta(days=1), periods=n_future)
    future = []
    for dt in future_dates:
        tmp = add_indicators(ext)
        win = sc_all.transform(tmp[features].values[-window:])
        p_s = model.predict(win.reshape(1, window, len(features)), verbose=0)[0, 0]
        p = float(sc_close.inverse_transform([[p_s]])[0, 0])
        future.append(p)
        ext.loc[dt] = [p, p, p, p, ext["Volume"].iloc[-1]]
    future = np.array(future)
    steps = np.arange(1, n_future + 1)
    band = 1.96 * sigma * np.sqrt(steps)

    names = ", ".join(["Giá đóng cửa"] + [FEAT_NAMES[f] for f in features[1:]])
    return {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "test_dates": d.index[split:],
        "pred": pred,
        "hist_close": d["Close"],
        "future_dates": future_dates, "future": future,
        "lower": np.maximum(future - band, 0), "upper": future + band,
        "metrics": dict(rmse=rmse, rmse_naive=rmse_naive, mae=mae, mape=mape, r2=r2, dir_acc=dir_acc),
        "info": f"AI học từ: {names} | nhìn lại {window} ngày | {len(d)} ngày dữ liệu",
        "last_date": d.index[-1],
    }


# ---- CHART HELPERS START ----
XD = alt.X("date:T", title=None)


def yq(field):
    return alt.Y(f"{field}:Q", scale=alt.Scale(zero=False), title=f"Giá{UNIT_TXT}",
                 axis=alt.Axis(format=FMT))


def show(chart):
    try:
        st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart)


def plot_df(df):
    """Đổi sang đơn vị hiển thị và chuẩn bị cột tên ASCII cho biểu đồ."""
    p = df.copy()
    price_cols = [c for c in ["Open", "High", "Low", "Close", "MA20", "MA50",
                              "BB_up", "BB_low", "MACD", "Signal", "Hist"] if c in p.columns]
    p[price_cols] = p[price_cols] * K
    p.index.name = "date"
    p = p.reset_index()
    p.columns = [c.lower() for c in p.columns]
    return p


def candle_chart(p, show_ma, show_bb):
    n = len(p)
    bar_size = int(max(2, min(10, 700 / max(n, 1) * 0.7)))
    up = alt.condition("datum.open <= datum.close", alt.value("#26a69a"), alt.value("#ef5350"))
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
        layers.append(base.mark_area(opacity=0.12, color="gray").encode(
            x=XD, y=yq("bb_low"), y2="bb_up:Q"))
    layers.append(base.mark_rule().encode(x=XD, y=yq("low"), y2="high:Q", color=up, tooltip=tip))
    layers.append(base.mark_bar(size=bar_size).encode(x=XD, y=yq("open"), y2="close:Q", color=up, tooltip=tip))
    if show_ma:
        ma = (base.transform_fold(["ma20", "ma50"], as_=["k", "v"])
              .transform_calculate(name="datum.k == 'ma20' ? 'TB 20 ngày' : 'TB 50 ngày'")
              .mark_line(strokeWidth=1.5)
              .encode(x=XD, y=yq("v"),
                      color=alt.Color("name:N", title=None,
                                      scale=alt.Scale(domain=["TB 20 ngày", "TB 50 ngày"],
                                                      range=["#ff9800", "#2196f3"])),
                      tooltip=[alt.Tooltip("date:T", title="Ngày"),
                               alt.Tooltip("name:N", title="Đường"),
                               alt.Tooltip("v:Q", title="Giá trị", format=FMT)]))
        layers.append(ma)
    return alt.layer(*layers).properties(height=380, width="container").interactive()


def volume_chart(p):
    up = alt.condition("datum.open <= datum.close", alt.value("#26a69a"), alt.value("#ef5350"))
    return (alt.Chart(p).mark_bar()
            .encode(x=XD,
                    y=alt.Y("volume:Q", title="Khối lượng (cổ phiếu)", axis=alt.Axis(format=",.0f")),
                    color=up,
                    tooltip=[alt.Tooltip("date:T", title="Ngày"),
                             alt.Tooltip("volume:Q", title="Khối lượng", format=",.0f")])
            .properties(height=140, width="container"))


def rsi_chart(p):
    line = (alt.Chart(p).mark_line()
            .encode(x=XD, y=alt.Y("rsi:Q", title="RSI", scale=alt.Scale(domain=[0, 100])),
                    tooltip=[alt.Tooltip("date:T", title="Ngày"),
                             alt.Tooltip("rsi:Q", title="RSI", format=".1f")]))
    rules = (alt.Chart(pd.DataFrame({"y": [30, 70], "c": ["#2e7d32", "#c62828"]}))
             .mark_rule(strokeDash=[5, 5])
             .encode(y="y:Q", color=alt.Color("c:N", scale=None, legend=None)))
    return alt.layer(line, rules).properties(height=240, width="container").interactive()


def macd_chart(p):
    hist = (alt.Chart(p).mark_bar(opacity=0.5)
            .encode(x=XD, y=alt.Y("hist:Q", title=f"MACD{UNIT_TXT}", axis=alt.Axis(format=FMT)),
                    color=alt.condition("datum.hist >= 0", alt.value("#26a69a"), alt.value("#ef5350")),
                    tooltip=[alt.Tooltip("date:T", title="Ngày"),
                             alt.Tooltip("hist:Q", title="Chênh lệch", format=FMT)]))
    lines = (alt.Chart(p).transform_fold(["macd", "signal"], as_=["k", "v"])
             .transform_calculate(name="datum.k == 'macd' ? 'MACD' : 'Đường tín hiệu'")
             .mark_line(strokeWidth=1.5)
             .encode(x=XD, y=alt.Y("v:Q", title=f"MACD{UNIT_TXT}", axis=alt.Axis(format=FMT)),
                     color=alt.Color("name:N", title=None,
                                     scale=alt.Scale(domain=["MACD", "Đường tín hiệu"],
                                                     range=["#2196f3", "#ff9800"])),
                     tooltip=[alt.Tooltip("date:T", title="Ngày"),
                              alt.Tooltip("name:N", title="Đường"),
                              alt.Tooltip("v:Q", title="Giá trị", format=FMT)]))
    return alt.layer(hist, lines).properties(height=260, width="container").interactive()


def forecast_chart(results):
    names = list(results.keys())
    first = results[names[0]]
    hist = first["hist_close"].tail(250)
    rows = [pd.DataFrame({"date": hist.index, "price": hist.values * K, "type": "Giá thật"})]
    pts, bands = [], []
    domain, rng = ["Giá thật"], ["#1f77b4"]
    for name, r in results.items():
        t_back = f"{name}: AI đoán lại quá khứ"
        t_fut = f"{name}: dự báo"
        last_p = float(r["hist_close"].iloc[-1])
        fd = pd.DatetimeIndex([r["last_date"]]).append(pd.DatetimeIndex(r["future_dates"]))
        rows.append(pd.DataFrame({"date": r["test_dates"], "price": r["pred"] * K, "type": t_back}))
        rows.append(pd.DataFrame({"date": fd, "price": np.concatenate([[last_p], r["future"]]) * K,
                                  "type": t_fut}))
        pts.append(pd.DataFrame({"date": r["future_dates"], "price": r["future"] * K, "type": t_fut}))
        bands.append(pd.DataFrame({
            "date": fd,
            "lower": np.concatenate([[last_p], r["lower"]]) * K,
            "upper": np.concatenate([[last_p], r["upper"]]) * K,
            "model": name}))
        domain += [t_back, t_fut]
        rng += [MODEL_LIGHT[name], MODEL_COLORS[name]]

    lines_df = pd.concat(rows, ignore_index=True)
    pts_df = pd.concat(pts, ignore_index=True)
    band_df = pd.concat(bands, ignore_index=True)
    color = alt.Color("type:N", title=None, scale=alt.Scale(domain=domain, range=rng))
    tip = [alt.Tooltip("date:T", title="Ngày"),
           alt.Tooltip("type:N", title="Loại"),
           alt.Tooltip("price:Q", title=f"Giá{UNIT_TXT}", format=FMT)]

    band = (alt.Chart(band_df).mark_area(opacity=0.18)
            .encode(x=XD, y=yq("lower"), y2="upper:Q",
                    color=alt.Color("model:N", legend=None,
                                    scale=alt.Scale(domain=names, range=[MODEL_COLORS[m] for m in names]))))
    line = alt.Chart(lines_df).mark_line(strokeWidth=2).encode(x=XD, y=yq("price"), color=color)
    hover = (alt.Chart(lines_df).mark_circle(size=70, opacity=0)
             .encode(x=XD, y=yq("price"), color=color, tooltip=tip))
    points = (alt.Chart(pts_df).mark_point(filled=True, size=55)
              .encode(x=XD, y=yq("price"), color=color, tooltip=tip))
    today = (alt.Chart(pd.DataFrame({"date": [first["last_date"]]}))
             .mark_rule(strokeDash=[4, 4], color="gray").encode(x=XD))
    return alt.layer(band, line, hover, points, today).properties(
        height=430, width="container").interactive()


def loss_chart(results):
    rows = []
    for name, r in results.items():
        n = len(r["loss"])
        rows.append(pd.DataFrame({"lan": range(1, n + 1), "v": r["loss"], "k": f"{name}: sai số khi học"}))
        rows.append(pd.DataFrame({"lan": range(1, n + 1), "v": r["val_loss"], "k": f"{name}: sai số khi tự kiểm tra"}))
    df = pd.concat(rows, ignore_index=True)
    return (alt.Chart(df).mark_line()
            .encode(x=alt.X("lan:Q", title="Lần học"), y=alt.Y("v:Q", title="Sai số"),
                    color=alt.Color("k:N", title=None),
                    tooltip=[alt.Tooltip("k:N", title="Loại"), alt.Tooltip("lan:Q", title="Lần học"),
                             alt.Tooltip("v:Q", title="Sai số", format=".5f")])
            .properties(height=250, width="container"))


def compare_chart(norm):
    n = norm.copy()
    n.index.name = "date"
    long = n.reset_index().melt(id_vars="date", var_name="ma", value_name="v")
    return (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=XD, y=alt.Y("v:Q", scale=alt.Scale(zero=False), title="Điểm (ngày đầu = 100)"),
                    color=alt.Color("ma:N", title="Mã"),
                    tooltip=[alt.Tooltip("date:T", title="Ngày"), alt.Tooltip("ma:N", title="Mã"),
                             alt.Tooltip("v:Q", title="Điểm", format=".1f")])
            .properties(height=380, width="container").interactive())


def corr_chart(corr):
    c = corr.copy()
    c.index.name = "a"
    c.columns.name = "b"
    long = c.reset_index().melt(id_vars="a", var_name="b", value_name="v")
    base = alt.Chart(long).encode(x=alt.X("b:N", title=None), y=alt.Y("a:N", title=None))
    rect = base.mark_rect().encode(
        color=alt.Color("v:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue"), title="Tương quan"),
        tooltip=[alt.Tooltip("a:N", title="Mã 1"), alt.Tooltip("b:N", title="Mã 2"),
                 alt.Tooltip("v:Q", title="Tương quan", format=".2f")])
    text = base.mark_text().encode(
        text=alt.Text("v:Q", format=".2f"),
        color=alt.condition("abs(datum.v) > 0.6", alt.value("white"), alt.value("black")))
    return (rect + text).properties(height=280, width="container")
# ---- CHART HELPERS END ----


# =============== DỮ LIỆU ===============
df_raw = load_data(ticker, start_date, end_date)
if df_raw.empty or len(df_raw) < 300:
    st.error("Không tải được đủ dữ liệu (cần ít nhất khoảng 300 ngày giao dịch). "
             "Kiểm tra lại mã cổ phiếu hoặc chọn khoảng thời gian dài hơn.")
    st.stop()
df_ind = add_indicators(df_raw)

# =============== NÚT DỰ BÁO (đặt ngoài tab để không bị nhảy tab) ===============
st.markdown(f"**Mã đang chọn:** {ticker}  |  **Dữ liệu:** {df_raw.index[0].date()} → {df_raw.index[-1].date()}")
if st.button("🚀 Bấm để AI học và dự báo", type="primary"):
    if not model_choices:
        st.warning("Hãy chọn ít nhất 1 kiểu AI ở thanh bên.")
    else:
        features = ["Close"] + [f for f in extra_feats if f != "Close"]
        try:
            results = {}
            for mname in model_choices:
                with st.spinner(f"AI ({mname}) đang học dữ liệu, vui lòng chờ 1–3 phút..."):
                    results[mname] = train_and_forecast(
                        df_raw, features, window_size, epochs, n_future, mname, seed)
            st.session_state["res"] = results
            st.session_state["res_ticker"] = ticker
            st.success("✅ Xong! Mở tab **🤖 Dự báo AI** bên dưới để xem kết quả.")
        except Exception as e:
            st.error(f"Không chạy được: {e}")
            st.code(traceback.format_exc())

tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan giá", "📐 Chỉ báo kỹ thuật", "🤖 Dự báo AI", "🔀 So sánh mã"])

# =============== TAB 1: TỔNG QUAN ===============
with tab1:
    last, prev_close = df_raw["Close"].iloc[-1], df_raw["Close"].iloc[-2]
    last_year = df_raw.tail(252)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá đóng cửa gần nhất", money(last), f"{(last / prev_close - 1) * 100:.2f}%")
    c2.metric("Giá cao nhất 1 năm", money(last_year["High"].max()))
    c3.metric("Giá thấp nhất 1 năm", money(last_year["Low"].min()))
    c4.metric("Khối lượng TB 20 ngày", f"{df_raw['Volume'].tail(20).mean():,.0f} cp")

    show_n = st.slider("Số ngày hiển thị trên biểu đồ", 60, min(len(df_ind), 500), 120, 10)
    show_ma = st.checkbox("Hiện đường trung bình 20 & 50 ngày", True)
    show_bb = st.checkbox("Hiện dải Bollinger (vùng giá dao động bình thường)", False)
    p = plot_df(df_ind.tail(show_n))

    st.caption("Nến xanh: ngày giá tăng, nến đỏ: ngày giá giảm. Rê/chạm vào biểu đồ để xem số liệu, "
               "cuộn hoặc kéo để phóng to/thu nhỏ.")
    show(candle_chart(p, show_ma, show_bb))
    show(volume_chart(p))

    with st.expander("📄 Xem & tải dữ liệu"):
        st.dataframe(df_raw.tail(30).round(2))
        st.download_button("⬇️ Tải toàn bộ dữ liệu (CSV)",
                           df_raw.to_csv().encode("utf-8-sig"), f"{ticker}_data.csv", "text/csv")

# =============== TAB 2: CHỈ BÁO ===============
with tab2:
    v = df_ind.dropna().tail(250)
    rsi_now = v["RSI"].iloc[-1]
    macd_now, sig_now = v["MACD"].iloc[-1], v["Signal"].iloc[-1]

    st.markdown("**Nhận định nhanh hiện tại**")
    if rsi_now > 70:
        st.warning(f"RSI = {rsi_now:.1f} → giá đang tăng nóng (trên 70), có thể sắp điều chỉnh.")
    elif rsi_now < 30:
        st.success(f"RSI = {rsi_now:.1f} → giá đang giảm sâu (dưới 30), có thể sắp hồi.")
    else:
        st.info(f"RSI = {rsi_now:.1f} → ở mức bình thường (30–70).")
    st.info("MACD đang " + ("**trên** đường tín hiệu → đà tăng đang mạnh hơn." if macd_now > sig_now
                            else "**dưới** đường tín hiệu → đà giảm đang mạnh hơn."))

    pv = plot_df(v)
    st.markdown("**RSI – sức mạnh giá** (trên 70: tăng nóng, dưới 30: giảm sâu)")
    show(rsi_chart(pv))
    st.markdown("**MACD – đà tăng/giảm** (MACD cắt lên đường tín hiệu: đà tăng)")
    show(macd_chart(pv))

# =============== TAB 3: DỰ BÁO AI ===============
with tab3:
    res = st.session_state.get("res")
    if not res:
        st.info("Chưa có kết quả. Chỉnh cấu hình ở thanh bên (nếu muốn) rồi bấm nút "
                "**🚀 Bấm để AI học và dự báo** ở phía trên.")
    else:
        names = list(res.keys())
        first = res[names[0]]
        last_close = float(first["hist_close"].iloc[-1])
        end_day = first["future_dates"][-1].strftime("%d/%m/%Y")

        st.caption(f"Kết quả của **{st.session_state['res_ticker']}** – {first['info']}")
        st.metric(f"Giá đóng cửa gần nhất ({first['last_date'].strftime('%d/%m/%Y')})", money(last_close))

        # ----- Các kịch bản dự báo -----
        st.markdown(f"### 🔮 Dự báo đến ngày {end_day}")
        for name in names:
            r = res[name]
            end_p = float(r["future"][-1])
            chg = (end_p / last_close - 1) * 100
            st.markdown(f"**Mô hình {name}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Kịch bản chính (AI cho là hợp lý nhất)", money(end_p), f"{chg:+.2f}% so với hôm nay")
            c2.metric("Kịch bản thấp (xấu)", money(float(r["lower"][-1])),
                      f"{(r['lower'][-1] / last_close - 1) * 100:+.2f}%")
            c3.metric("Kịch bản cao (tốt)", money(float(r["upper"][-1])),
                      f"{(r['upper'][-1] / last_close - 1) * 100:+.2f}%")

        st.caption("Cách xem: đường đậm là dự báo, vùng tô màu là khoảng giá có thể dao động (~95%). "
                   "Đường nét chấm mờ là AI thử đoán lại quá khứ để kiểm tra. "
                   "Rê/chạm vào biểu đồ để xem số, cuộn hoặc kéo để phóng to.")
        show(forecast_chart(res))

        # ----- Bảng dự báo từng ngày -----
        st.markdown("### 📅 Bảng dự báo từng ngày")
        disp = pd.DataFrame({"Ngày": [d.strftime("%d/%m/%Y") for d in first["future_dates"]]})
        num = pd.DataFrame({"Ngày": first["future_dates"].date})
        for name in names:
            r = res[name]
            base_p = float(r["hist_close"].iloc[-1])
            cols = {
                f"{name}: dự báo{UNIT_TXT}": r["future"] * K,
                f"{name}: thấp nhất có thể{UNIT_TXT}": r["lower"] * K,
                f"{name}: cao nhất có thể{UNIT_TXT}": r["upper"] * K,
                f"{name}: so với hôm nay (%)": (r["future"] / base_p - 1) * 100,
            }
            for col, vals in cols.items():
                num[col] = np.round(vals, 2)
                if "(%)" in col:
                    disp[col] = [f"{x:+.2f}%" for x in vals]
                else:
                    disp[col] = [f"{x:,.{DEC}f}" for x in vals]
        st.dataframe(disp, hide_index=True)
        st.download_button("⬇️ Tải bảng dự báo (CSV)", num.to_csv(index=False).encode("utf-8-sig"),
                           f"{ticker}_du_bao.csv", "text/csv")

        # ----- Độ chính xác -----
        st.markdown("### 🎯 AI đoán chính xác đến đâu?")
        st.caption("Chấm trên 20% dữ liệu mới nhất mà AI chưa từng học.")
        acc_rows = []
        for name in names:
            m = res[name]["metrics"]
            acc_rows.append({
                "Mô hình": name,
                f"Sai số trung bình (RMSE){UNIT_TXT}": f"{m['rmse'] * K:,.{DEC}f}",
                f"Sai lệch trung bình (MAE){UNIT_TXT}": f"{m['mae'] * K:,.{DEC}f}",
                "Sai lệch theo %": f"{m['mape']:.2f}%",
                "Độ khớp (R²)": f"{m['r2']:.3f}",
                "Đoán đúng tăng/giảm": f"{m['dir_acc']:.1f}%",
                f"Cách đơn giản 'mai = hôm nay'{UNIT_TXT}": f"{m['rmse_naive'] * K:,.{DEC}f}",
                "Kết luận": "Tốt hơn cách đơn giản ✅" if m["rmse"] < m["rmse_naive"]
                else "Chưa tốt hơn cách đơn giản ⚠️",
            })
        st.dataframe(pd.DataFrame(acc_rows), hide_index=True)

        with st.expander("📖 Giải thích các chỉ số"):
            st.markdown(
                "- **Sai số trung bình (RMSE):** AI đoán lệch khoảng bao nhiêu so với giá thật. Càng nhỏ càng tốt.\n"
                "- **Sai lệch trung bình (MAE):** mỗi ngày AI đoán lệch trung bình bao nhiêu tiền.\n"
                "- **Sai lệch theo %:** AI đoán lệch trung bình bao nhiêu % so với giá thật.\n"
                "- **Độ khớp (R²):** đường dự báo khớp giá thật đến đâu. Gần 1 là rất khớp, thấp hoặc âm là kém.\n"
                "- **Đoán đúng tăng/giảm:** tỷ lệ ngày AI đoán đúng giá tăng hay giảm. Khoảng 50% nghĩa là như tung đồng xu.\n"
                "- **Cách đơn giản 'mai = hôm nay':** cứ lấy giá hôm nay làm giá ngày mai. "
                "Nếu AI không thắng được cách này thì dự báo chưa đáng tin.")

        with st.expander("📉 Quá trình AI học (dành cho người muốn xem chi tiết)"):
            st.caption("Hai đường càng thấp và sát nhau càng tốt.")
            show(loss_chart(res))

        st.caption("⚠️ Vùng dao động chỉ là ước lượng gần đúng. Dự báo chỉ dùng để học tập, "
                   "không phải lời khuyên đầu tư.")

# =============== TAB 4: SO SÁNH MÃ ===============
with tab4:
    tickers_text = st.text_input("Nhập các mã muốn so sánh, cách nhau bằng dấu phẩy",
                                 "VNM.VN, FPT.VN, HPG.VN")
    cmp_start = st.date_input("So sánh từ ngày", max(start_date, end_date - timedelta(days=365 * 2)),
                              key="cmp_start")
    tks = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    closes = {}
    for t in tks:
        try:
            dft = load_data(t, cmp_start, end_date)
            if not dft.empty:
                closes[t] = dft["Close"]
        except Exception:
            st.warning(f"Không tải được {t}")

    if len(closes) >= 2:
        cdf = pd.DataFrame(closes).dropna()
        norm = cdf / cdf.iloc[0] * 100

        st.markdown("**Mã nào tăng tốt hơn?** (tất cả quy về 100 điểm tại ngày bắt đầu)")
        show(compare_chart(norm))

        summary = pd.DataFrame({
            "Lãi/lỗ cả giai đoạn": [f"{x:+.2f}%" for x in (cdf.iloc[-1] / cdf.iloc[0] - 1) * 100],
            "Mức dao động giá mỗi năm": [f"{x:.2f}%" for x in cdf.pct_change().std() * np.sqrt(252) * 100],
        }, index=cdf.columns)
        st.dataframe(summary)
        st.caption("Mức dao động càng cao thì giá lên xuống càng mạnh, rủi ro càng lớn.")

        st.markdown("**Các mã có tăng giảm cùng nhau không?** (gần 1: cùng chiều, gần 0: không liên quan)")
        show(corr_chart(cdf.pct_change().dropna().corr()))
    else:
        st.info("Nhập ít nhất 2 mã hợp lệ để so sánh.")
