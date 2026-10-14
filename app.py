import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from keras.callbacks import EarlyStopping
from keras.layers import GRU, LSTM, Dense, Dropout, Input
from keras.models import Sequential
from plotly.subplots import make_subplots
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
        epochs=epochs, batch_size=32, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)],
    )

    # ----- Đánh giá trên tập test -----
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

    # ----- Dự báo tương lai (đệ quy, tính lại chỉ báo mỗi bước) -----
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

    return {
        "loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "test_dates": d.index[split:],
        "real": real, "pred": pred,
        "hist_close": d["Close"],
        "future_dates": future_dates, "future": future,
        "lower": future - band, "upper": future + band,
        "metrics": dict(rmse=rmse, rmse_naive=rmse_naive, mae=mae, mape=mape, r2=r2, dir_acc=dir_acc),
        "info": f"{model_type} | đặc trưng: {', '.join(features)} | window {window} | {len(d)} phiên",
    }


# =============== DỮ LIỆU ===============
df_raw = load_data(ticker, start_date)
if df_raw.empty or len(df_raw) < 300:
    st.error("Không tải được đủ dữ liệu. Kiểm tra mã cổ phiếu hoặc chọn ngày bắt đầu sớm hơn.")
    st.stop()
df_ind = add_indicators(df_raw)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Tổng quan", "📐 Chỉ báo kỹ thuật", "🤖 Dự báo AI", "🔀 So sánh mã"])

# =============== TAB 1: TỔNG QUAN ===============
with tab1:
    last, prev_close = df_raw["Close"].iloc[-1], df_raw["Close"].iloc[-2]
    last_year = df_raw.tail(252)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá đóng cửa", f"{last:,.2f}", f"{(last / prev_close - 1) * 100:.2f}%")
    c2.metric("Cao nhất 52 tuần", f"{last_year['High'].max():,.2f}")
    c3.metric("Thấp nhất 52 tuần", f"{last_year['Low'].min():,.2f}")
    c4.metric("KL trung bình 20 phiên", f"{df_raw['Volume'].tail(20).mean():,.0f}")

    show_n = st.slider("Số phiên hiển thị", 60, min(len(df_ind), 1500), 250, 10)
    show_ma = st.checkbox("Hiện MA20 / MA50", True)
    show_bb = st.checkbox("Hiện Bollinger Bands", False)
    v = df_ind.tail(show_n)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=v.index, open=v["Open"], high=v["High"], low=v["Low"],
                                 close=v["Close"], name="Giá"), row=1, col=1)
    if show_ma:
        fig.add_trace(go.Scatter(x=v.index, y=v["MA20"], name="MA20", line=dict(width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=v["MA50"], name="MA50", line=dict(width=1.2)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=v.index, y=v["BB_up"], name="BB trên",
                                 line=dict(width=0.8, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=v["BB_low"], name="BB dưới",
                                 line=dict(width=0.8, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=v.index, y=v["Volume"], name="Khối lượng"), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      title=f"{ticker} – nến Nhật & khối lượng")
    st.plotly_chart(fig)

    with st.expander("📄 Xem & tải dữ liệu"):
        st.dataframe(df_raw.tail(30).round(2))
        st.download_button("⬇️ Tải toàn bộ dữ liệu (CSV)",
                           df_raw.to_csv().encode("utf-8-sig"), f"{ticker}_data.csv", "text/csv")

# =============== TAB 2: CHỈ BÁO ===============
with tab2:
    v = df_ind.dropna().tail(250)
    rsi_now = v["RSI"].iloc[-1]
    macd_now, sig_now = v["MACD"].iloc[-1], v["Signal"].iloc[-1]

    st.markdown("**Tín hiệu hiện tại**")
    if rsi_now > 70:
        st.warning(f"RSI = {rsi_now:.1f} → vùng quá mua")
    elif rsi_now < 30:
        st.success(f"RSI = {rsi_now:.1f} → vùng quá bán")
    else:
        st.info(f"RSI = {rsi_now:.1f} → vùng trung tính")
    st.info("MACD đang " + ("**trên** đường Signal (động lực tăng)" if macd_now > sig_now
                            else "**dưới** đường Signal (động lực giảm)"))

    fig_r = go.Figure(go.Scatter(x=v.index, y=v["RSI"], name="RSI"))
    fig_r.add_hline(y=70, line_dash="dash", line_color="red")
    fig_r.add_hline(y=30, line_dash="dash", line_color="green")
    fig_r.update_layout(height=280, title="RSI (14)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_r)

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(x=v.index, y=v["Hist"], name="Histogram"))
    fig_m.add_trace(go.Scatter(x=v.index, y=v["MACD"], name="MACD"))
    fig_m.add_trace(go.Scatter(x=v.index, y=v["Signal"], name="Signal"))
    fig_m.update_layout(height=300, title="MACD (12, 26, 9)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_m)

# =============== TAB 3: DỰ BÁO AI ===============
with tab3:
    st.write("Cấu hình mô hình ở thanh bên, sau đó bấm nút dưới đây.")
    if st.button("🚀 Huấn luyện & dự báo", type="primary"):
        features = ["Close"] + [f for f in extra_feats if f != "Close"]
        with st.spinner("Đang huấn luyện mô hình (có thể mất 1–3 phút)..."):
            st.session_state["res"] = train_and_forecast(
                df_raw, features, window_size, epochs, n_future, model_type, seed)
            st.session_state["res_ticker"] = ticker

    res = st.session_state.get("res")
    if res is not None:
        st.caption(f"Kết quả của **{st.session_state['res_ticker']}** – {res['info']}")
        m = res["metrics"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE", f"{m['rmse']:,.2f}")
        c2.metric("MAE", f"{m['mae']:,.2f}")
        c3.metric("MAPE", f"{m['mape']:.2f}%")
        c4.metric("R²", f"{m['r2']:.3f}")
        c5.metric("Đúng hướng", f"{m['dir_acc']:.1f}%")

        if m["rmse"] < m["rmse_naive"]:
            st.success(f"Mô hình tốt hơn baseline 'giá hôm qua' (RMSE {m['rmse']:,.2f} < {m['rmse_naive']:,.2f}).")
        else:
            st.warning(f"Mô hình chưa vượt baseline 'giá hôm qua' (RMSE {m['rmse']:,.2f} ≥ {m['rmse_naive']:,.2f}). "
                       "Thử đổi đặc trưng, window hoặc epochs.")

        # ----- Biểu đồ dự báo -----
        hist = res["hist_close"].tail(250)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Giá thật"))
        fig.add_trace(go.Scatter(x=res["test_dates"], y=res["pred"], name="Dự báo (test)",
                                 line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=res["future_dates"], y=res["upper"], line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=res["future_dates"], y=res["lower"], line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,127,14,0.2)",
                                 name="Khoảng tin cậy ~95%"))
        fig.add_trace(go.Scatter(x=res["future_dates"], y=res["future"], mode="lines+markers",
                                 name=f"Dự báo {len(res['future'])} ngày"))
        fig.update_layout(height=500, title="Giá thật vs dự báo", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            future_df = pd.DataFrame({
                "Ngày": res["future_dates"].date,
                "Dự báo": res["future"].round(2),
                "Thấp (95%)": res["lower"].round(2),
                "Cao (95%)": res["upper"].round(2),
            })
            st.markdown("**Bảng dự báo**")
            st.dataframe(future_df, hide_index=True)
            st.download_button("⬇️ Tải dự báo (CSV)", future_df.to_csv(index=False).encode("utf-8-sig"),
                               f"{ticker}_du_bao.csv", "text/csv")
        with col_b:
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(y=res["loss"], name="Train loss"))
            fig_l.add_trace(go.Scatter(y=res["val_loss"], name="Validation loss"))
            fig_l.update_layout(height=320, title="Loss khi huấn luyện", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_l)

        st.caption("⚠️ Khoảng tin cậy chỉ là ước lượng gần đúng từ sai số trên tập test. "
                   "Chỉ dùng cho học tập, không phải khuyến nghị đầu tư.")

# =============== TAB 4: SO SÁNH MÃ ===============
with tab4:
    tickers_text = st.text_input("Nhập các mã, cách nhau bằng dấu phẩy", "VNM.VN, FPT.VN, HPG.VN")
    cmp_start = st.date_input("So sánh từ ngày", date.today() - timedelta(days=365 * 2), key="cmp_start")
    tks = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    if st.button("So sánh"):
        closes = {}
        for t in tks:
            try:
                dft = load_data(t, cmp_start)
                if not dft.empty:
                    closes[t] = dft["Close"]
            except Exception:
                st.warning(f"Không tải được {t}")
        if len(closes) >= 2:
            cdf = pd.DataFrame(closes).dropna()
            norm = cdf / cdf.iloc[0] * 100
            fig_c = go.Figure([go.Scatter(x=norm.index, y=norm[c], name=c) for c in norm.columns])
            fig_c.update_layout(height=420, title="Hiệu suất (quy về 100 tại ngày bắt đầu)",
                                margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_c)

            summary = pd.DataFrame({
                "Lợi nhuận (%)": (cdf.iloc[-1] / cdf.iloc[0] - 1) * 100,
                "Biến động năm (%)": cdf.pct_change().std() * np.sqrt(252) * 100,
            }).round(2)
            st.dataframe(summary)

            corr = cdf.pct_change().dropna().corr().round(2)
            fig_h = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                         text=corr.values, texttemplate="%{text}",
                                         zmin=-1, zmax=1, colorscale="RdBu"))
            fig_h.update_layout(height=380, title="Tương quan lợi suất ngày", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_h)
        else:
            st.error("Cần ít nhất 2 mã hợp lệ để so sánh.")
