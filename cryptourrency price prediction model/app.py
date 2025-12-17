import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from datetime import date
import streamlit.components.v1 as components

COIN_LAUNCH_DATES = {
    "Bitcoin (BTC)": date(2009, 1, 3),
    "Ethereum (ETH)": date(2015, 7, 30),
    "Binancecoin (BNB)": date(2017, 7, 25),
    "Ripple (XRP)": date(2012, 6, 2),
    "Dogecoin(DOGE)": date(2013,12,6)
   
}



# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crypto Price Prediction (LSTM)",
    page_icon="₿",
    layout="centered"
)


# ---------------- BACKGROUND ----------------
def add_bg_from_local(image_file):
    import base64
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
    <style>
    
        /* Full height reset */
        html, body, [class*="css"] {{
        
    overscroll-behavior: none;



            height: 100%;
            margin: 0;
            padding: 0;
            
        }}

        /* Main app background */
    .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom:0;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
        }}

        /* REMOVE TOP WHITE GAP */
        header {{
            display: none !important;
        }}

        section.main {{
            padding-top: 0px !important;
        }}
       section.main > div {{
            padding-top: 0px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
)
    # ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style="
        text-align:center;
        color:#7C4DFF;
        text-shadow: 2px 2px 10px black;
    ">
        Cryptocurrency Price Prediction (LSTM):
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
/* Label text color */
label {
    color: #00FFD2 !important;
    font-weight: 600;
}

/* Selectbox text */
div[data-baseweb="select"] span {
    color: #000000;
}

/* Date input text */
input {
    color: #000000 !important;
}

/* Slider label */
div[data-testid="stSlider"] label {
    color: #B388FF !important;
}
</style>
""", unsafe_allow_html=True)


    

add_bg_from_local("Bitcoin-Price-Prediction-2023-735x400.png")

# ---------------- INPUTS (MAIN PAGE ONLY) ----------------
st.markdown(
    "<h3 style='color:#FF1744;'>🔧 Select Inputs</h3>",
    unsafe_allow_html=True
)

crypto = st.selectbox(
    "Select Cryptocurrency",
    ["Bitcoin (BTC)", "Ethereum (ETH)","Binancecoin (BNB)","Ripple (XRP)","Dogecoin(DOGE)"]
)

start_date = st.date_input("From Date", value=date(2022, 1, 1))
end_date = st.date_input("To Date", value=date.today())


prediction_days = st.sidebar.slider(
    "Prediction Interval (Days)", 1, 30, 7
)


show_prediction = st.sidebar.checkbox("Show Prediction", value=True)

interval = st.selectbox(
    "Prediction Interval",
    ["1 Day", "7 Days", "30 Days"]
)
launch_date = COIN_LAUNCH_DATES.get(crypto)

launch_date = COIN_LAUNCH_DATES.get(crypto)

if start_date < launch_date:
    st.markdown(
        f"""
        <div style="
            background:rgba(0,0,0,0.75);
            border-left:6px solid #ff1744;
            padding:20px;
            border-radius:12px;
            color:#ffb4b4;
            font-size:22px;
            font-weight:700;
            text-align:center;
            line-height:1.6;
            box-shadow:0 0 15px rgba(255,23,68,0.6);
        ">
        🚨 <span style="font-size:26px;color:#ff5252;">
        {crypto}
        </span><br>
        was launched on <b>{launch_date}</b><br>
        ❗ Please select a valid start date
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

    
# ---------------- SYMBOL & MODEL MAP ----------------
symbol_model_map = {
    "Bitcoin (BTC)": {
        "symbol": "BTC-USD",
        "model": "model.h5"
    },
    "Ethereum (ETH)": {
        "symbol": "ETH-USD",
        "model": "eth_model.h5"
    },
    "Binancecoin (BNB)": {
        "symbol": "BNB-USD",
        "model": "bnb_model.h5"
    },
    "Ripple (XRP)": {
        "symbol": "XRP-USD",
        "model": "xrp_model.h5"
    },
    "Dogecoin(DOGE)": {
        "symbol": "DOGE-USD",
        "model": "doge_model.h5"
    }
   
}

selected_symbol = symbol_model_map[crypto]["symbol"]
model_path = symbol_model_map[crypto]["model"]

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style="
        text-align:center;
        color:#00FFD1;
        text-shadow: 2px 2px 10px black;
    ">
    Cryptocurrency Price Prediction (LSTM) model:
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style="
        text-align:center;
        color:#FFD701;
        font-size:16px;
        font-weight:500;
        text-shadow: 1px 1px 6px black;
    ">
        Based on Yahoo Finance CRYPTO-USD data
    </p>
    """,
    unsafe_allow_html=True
)




# ---------------- CURRENT PRICE ----------------
ticker = yf.Ticker(selected_symbol)
current_price = ticker.history(period="1d")["Close"].iloc[-1]

st.markdown(f"""
<div style="background:rgba(0,0,0,0.6);padding:20px;border-radius:15px;
text-align:center;color:gold;font-size:28px;">
💰 Current Price<br><b>${current_price:,.2f}</b>
</div>
""", unsafe_allow_html=True)
st.write("CRYPTO VALUE:", repr(crypto))
st.write("MODEL PATH:", repr(model_path))

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

data = load_data(selected_symbol, start_date, end_date)


features = data[['Close']]

# ---------------- PRICE ANALYTICS ----------------
st.markdown("""
<div style="
    background:rgba(0,0,0,0.7);
    padding:15px 25px;
    border-radius:14px;
    display:inline-block;
    margin-top:20px;
    margin-bottom:15px;
    box-shadow:0 0 20px rgba(0,255,209,0.6);
">
    <span style="
        font-size:26px;
        font-weight:800;
        color:#00FFD1;
        letter-spacing:1px;
    ">
        📊 Price Analytics
    </span>
</div>
""", unsafe_allow_html=True)

min_price = float(data["Close"].min())
max_price = float(data["Close"].max())
avg_price = float(data["Close"].mean())

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div style="
    background:rgba(0,0,0,0.75);
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 0 15px rgba(0,255,209,0.5);
">
    <div style="color:#00FFD1;font-size:18px;font-weight:700;">
        📉 Min Price
    </div>
    <div style="color:red;font-size:26px;font-weight:800;">
        ${min_price:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div style="
    background:rgba(0,0,0,0.75);
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 0 15px rgba(255,215,0,0.6);
">
    <div style="color:#FFD700;font-size:18px;font-weight:700;">
        📈 Max Price
    </div>
    <div style="color:fuchsia;font-size:26px;font-weight:800;">
        ${max_price:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div style="
    background:rgba(0,0,0,0.75);
    padding:18px;
    border-radius:15px;
    text-align:center;
    box-shadow:0 0 15px rgba(179,136,255,0.6);
">
    <div style="color:#B388FF;font-size:18px;font-weight:700;">
        📊 Avg Price
    </div>
    <div style="color:pink;font-size:26px;font-weight:800;">
        ${avg_price:,.2f}
    </div>
</div>
""", unsafe_allow_html=True)
# ---------------- SCALING ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(model_path,
                       compile=False)
except:
    st.error(f"❌ Model not found: {model_path}")
    st.stop()

# ---------------- PREPARE INPUT ----------------
look_back = 60
X_test = []

for i in range(look_back, len(scaled_data)):
    X_test.append(scaled_data[i - look_back:i])

X_test = np.array(X_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
if len(data) < 60:
    st.error("Not enough historical data for LSTM prediction.")
    st.stop()

# ---------------- PREDICTION ----------------
predicted_scaled = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()

# ---------------- FUTURE DATES ----------------
last_date = data.index[-1]
future_dates = pd.date_range(
    start=last_date,
    periods=prediction_days + 1,
    freq="D"
)[1:]

future_predictions = predicted_prices[-prediction_days:]

# ---------------- PLOT ----------------
st.markdown("<h2 style='color:#FF4B4B;'>📈 Price Trend </h2>",
            unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, future_predictions, marker="o", label="Predicted Price")
ax.set_title(f"{crypto} Future Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)
# ---------------- FINAL FUTURE PREDICTED PRICE ----------------
final_predicted_price = float(future_predictions[-1])
# ---------------- FINAL FUTURE PREDICTED PRICE ----------------
final_predicted_price = float(future_predictions[-1])

trend = "UP 📈" if final_predicted_price > current_price else "DOWN 📉"
trend_color = "#00ff99" if trend.startswith("UP") else "#ff5252"

components.html(
    f"""
    <div style="
        background:rgba(0,0,0,0.75);
        padding:30px;
        border-radius:18px;
        margin-top:30px;
        text-align:center;
        box-shadow:0 0 30px {trend_color};
        font-family:Arial;
    ">
        <div style="font-size:22px;color:#00FFD1;">
            🔮 Final Predicted Price
        </div>

        <div style="font-size:40px;font-weight:800;color:#FFD700;">
            ${final_predicted_price:,.2f}
        </div>

        <div style="font-size:22px;color:{trend_color};margin-top:12px;">
            Trend Direction: {trend}
        </div>
    </div>
    """,
    height=220,
)

# ---------------- FINAL PRICE ----------------
future_price = float(predicted_prices[-1])

# ---------------- TREND DIRECTION ----------------
future_price = float(predicted_prices[-1])
current_price = float(current_price)


# ---------------- EXPORT ----------------
export_df = pd.DataFrame({
    "Date": data.index,
    "Actual_Price_USD": data["Close"].values.flatten()
})
export_df.to_csv(f"{selected_symbol}_price_history.csv", index=False)
