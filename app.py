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
        "lower": future - band, "upper": future + band,
        "metrics": dict(rmse=rmse, rmse_naive=rmse_naive, mae=mae, mape=mape, r2=r2, dir_acc=dir_acc),
        "info": f"Mô hình {model_type} | AI học từ: {names} | nhìn lại {window} ngày | {len(d)} ngày dữ liệu",
        "last_date": d.index[-1],
    }


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
    features = ["Close"] + [f for f in extra_feats if f != "Close"]
    try:
        with st.spinner("AI đang học dữ liệu, vui lòng chờ 1–3 phút..."):
            st.session_state["res"] = train_and_forecast(
                df_raw, features, window_size, epochs, n_future, model_type, seed)
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
    c1.metric("Giá đóng cửa gần nhất", f"{last:,.2f}", f"{(last / prev_close - 1) * 100:.2f}%")
    c2.metric("Giá cao nhất 1 năm", f"{last_year['High'].max():,.2f}")
    c3.metric("Giá thấp nhất 1 năm", f"{last_year['Low'].min():,.2f}")
    c4.metric("Khối lượng TB 20 ngày", f"{df_raw['Volume'].tail(20).mean():,.0f}")

    show_n = st.slider("Số ngày hiển thị trên biểu đồ", 60, min(len(df_ind), 500), 120, 10)
    show_ma = st.checkbox("Hiện đường trung bình 20 & 50 ngày", True)
    show_bb = st.checkbox("Hiện dải Bollinger (vùng giá dao động bình thường)", False)
    v = df_ind.tail(show_n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    draw_candles(ax1, v)
    if show_ma:
        ax1.plot(v.index, v["MA20"], label="TB 20 ngày", linewidth=1.2)
        ax1.plot(v.index, v["MA50"], label="TB 50 ngày", linewidth=1.2)
    if show_bb:
        ax1.plot(v.index, v["BB_up"], "--", label="Dải trên", linewidth=0.8)
        ax1.plot(v.index, v["BB_low"], "--", label="Dải dưới", linewidth=0.8)
    ax1.set_title(f"{ticker} – biểu đồ nến (xanh: ngày tăng, đỏ: ngày giảm)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.bar(v.index, v["Volume"], width=0.7, color="gray")
    ax2.set_title("Khối lượng giao dịch")
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)

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

    fig_r, ax = plt.subplots(figsize=(11, 3))
    ax.plot(v.index, v["RSI"])
    ax.axhline(70, color="red", linestyle="--")
    ax.axhline(30, color="green", linestyle="--")
    ax.set_title("RSI – sức mạnh giá (trên 70: tăng nóng, dưới 30: giảm sâu)")
    ax.grid(alpha=0.3)
    fig_r.tight_layout()
    st.pyplot(fig_r)

    fig_m, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(v.index, v["Hist"], width=0.7, color="gray", label="Chênh lệch")
    ax.plot(v.index, v["MACD"], label="MACD")
    ax.plot(v.index, v["Signal"], label="Đường tín hiệu")
    ax.set_title("MACD – đà tăng/giảm (MACD cắt lên đường tín hiệu: đà tăng)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig_m.tight_layout()
    st.pyplot(fig_m)

# =============== TAB 3: DỰ BÁO AI ===============
with tab3:
    res = st.session_state.get("res")
    if res is None:
        st.info("Chưa có kết quả. Chỉnh cấu hình ở thanh bên (nếu muốn) rồi bấm nút "
                "**🚀 Bấm để AI học và dự báo** ở phía trên.")
    else:
        st.caption(f"Kết quả của **{st.session_state['res_ticker']}** – {res['info']}")
        m = res["metrics"]

        st.markdown("**AI đoán chính xác đến đâu?** (chấm trên 20% dữ liệu mới nhất mà AI chưa từng học)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sai số trung bình", f"{m['rmse']:,.2f}",
                  help="RMSE: AI đoán lệch khoảng bao nhiêu so với giá thật. Càng nhỏ càng tốt.")
        c2.metric("Sai lệch trung bình", f"{m['mae']:,.2f}",
                  help="MAE: trung bình mỗi ngày AI đoán lệch bao nhiêu đồng. Càng nhỏ càng tốt.")
        c3.metric("Sai lệch theo %", f"{m['mape']:.2f}%",
                  help="MAPE: AI đoán lệch trung bình bao nhiêu % so với giá thật.")
        c4.metric("Độ khớp", f"{m['r2']:.3f}",
                  help="R²: đường dự báo khớp với giá thật đến đâu. Gần 1 là rất khớp, thấp hoặc âm là kém.")
        c5.metric("Đoán đúng tăng/giảm", f"{m['dir_acc']:.1f}%",
                  help="Tỷ lệ ngày AI đoán đúng giá sẽ tăng hay giảm. Khoảng 50% nghĩa là như tung đồng xu.")

        if m["rmse"] < m["rmse_naive"]:
            st.success(f"AI đoán tốt hơn cách đơn giản 'ngày mai bằng hôm nay' "
                       f"(sai số {m['rmse']:,.2f} so với {m['rmse_naive']:,.2f}).")
        else:
            st.warning(f"AI chưa đoán tốt hơn cách đơn giản 'ngày mai bằng hôm nay' "
                       f"(sai số {m['rmse']:,.2f} so với {m['rmse_naive']:,.2f}). "
                       "Hãy thử đổi thông tin thêm, số ngày nhìn lại hoặc số lần học.")

        # ----- Biểu đồ dự báo -----
        hist = res["hist_close"].tail(250)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(hist.index, hist.values, label="Giá thật")
        ax.plot(res["test_dates"], res["pred"], ":", label="AI đoán lại quá khứ (để kiểm tra)")
        ax.fill_between(res["future_dates"], res["lower"], res["upper"],
                        color="orange", alpha=0.25, label="Vùng giá có thể dao động (~95%)")
        ax.plot(res["future_dates"], res["future"], "o-", color="darkorange",
                label=f"Dự báo {len(res['future'])} ngày tới")
        ax.set_title("Giá thật và dự báo")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig)

        future_df = pd.DataFrame({
            "Ngày": res["future_dates"].date,
            "Giá dự báo": res["future"].round(2),
            "Thấp nhất có thể": res["lower"].round(2),
            "Cao nhất có thể": res["upper"].round(2),
        })
        st.markdown("**Bảng dự báo**")
        st.dataframe(future_df, hide_index=True)
        st.download_button("⬇️ Tải bảng dự báo (CSV)", future_df.to_csv(index=False).encode("utf-8-sig"),
                           f"{ticker}_du_bao.csv", "text/csv")

        with st.expander("📉 Quá trình AI học (dành cho người muốn xem chi tiết)"):
            fig_l, ax = plt.subplots(figsize=(11, 3.5))
            ax.plot(res["loss"], label="Sai số khi học")
            ax.plot(res["val_loss"], label="Sai số khi tự kiểm tra")
            ax.set_title("Hai đường càng thấp và sát nhau càng tốt")
            ax.legend()
            ax.grid(alpha=0.3)
            fig_l.tight_layout()
            st.pyplot(fig_l)

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

        fig_c, ax = plt.subplots(figsize=(11, 4.5))
        for c in norm.columns:
            ax.plot(norm.index, norm[c], label=c)
        ax.set_title("Mã nào tăng tốt hơn? (tất cả quy về 100 tại ngày bắt đầu)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig_c.autofmt_xdate()
        fig_c.tight_layout()
        st.pyplot(fig_c)

        summary = pd.DataFrame({
            "Lãi/lỗ cả giai đoạn (%)": (cdf.iloc[-1] / cdf.iloc[0] - 1) * 100,
            "Mức dao động giá mỗi năm (%)": cdf.pct_change().std() * np.sqrt(252) * 100,
        }).round(2)
        st.dataframe(summary)
        st.caption("Mức dao động càng cao thì giá lên xuống càng mạnh, rủi ro càng lớn.")

        corr = cdf.pct_change().dropna().corr()
        fig_h, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45)
        ax.set_yticklabels(corr.index)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
        ax.set_title("Các mã có tăng giảm cùng nhau không?\n(gần 1: cùng chiều, gần 0: không liên quan)")
        fig_h.colorbar(im)
        fig_h.tight_layout()
        st.pyplot(fig_h)
    else:
        st.info("Nhập ít nhất 2 mã hợp lệ để so sánh.")
