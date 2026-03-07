import streamlit as st
# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Price Prediction (LSTM)",
    page_icon="₿",
    layout="wide"
)
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta
import streamlit.components.v1 as components
from auth import create_users_table, register_user, authenticate_user
create_users_table()
import time


# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ✅ ADD THIS
if "login_mode" not in st.session_state:
    st.session_state.login_mode = "login"

COIN_LAUNCH_DATES = {
    "Bitcoin (BTC)": date(2009, 1, 3),
    "Ethereum (ETH)": date(2015, 7, 30),
    "Binancecoin (BNB)": date(2017, 7, 25),
    "Ripple (XRP)": date(2012, 6, 2),
    "Dogecoin(DOGE)": date(2013,12,6)
}


# --- Background and Gradient Function ---
def set_page_background(image_file=None, gradient_start=None, gradient_end=None):
    import base64

    background_style_property = ""
    if image_file:
        try:
            with open(image_file, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode()
            background_style_property = f"background-image: url(\"data:image/png;base64,{encoded_image}\");"
        except FileNotFoundError:
            st.error(f"Background image file not found: {image_file}")
    elif gradient_start and gradient_end:
        background_style_property = f"background: linear-gradient(to bottom right, {gradient_start}, {gradient_end});"

    style_css = f"""
    <style>
        html, body, .stApp {{ 
            {background_style_property}
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            overscroll-behavior: none;
            height: 100%;
            margin: 0;
            padding: 0;
        }}
        header {{
            display: none !important;
        }}
        section.main {{
            padding-top: 0px !important;
        }}
        section.main > div {{
            padding-top: 0px !important;
        }}
        .crypto-card {{
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease-in-out;
            height: 200px; 
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .crypto-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
        }}
        .crypto-name {{
            font-size: 24px;
            font-weight: bold;
            color: #00FFD1;
            margin-bottom: 10px;
        }}
        .crypto-price {{
            font-size: 32px;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 5px;
        }}
        .price-change {{
            font-size: 18px;
            font-weight: 600;
        }}
        .positive-change {{
            color: #00FF99;
        }}
        .negative-change {{
            color: #FF5252;
        }}
    </style>
    """
    st.markdown(style_css, unsafe_allow_html=True)

# --- Custom Styles (retained from original) ---
st.markdown("""
<style>
/* Label text color */
label {
    color: #1E90FF !important; /* Blue color for labels in login */
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

</style>
""", unsafe_allow_html=True)



# --- Login Page ---
# ---------------- LOGIN PAGE ----------------
def login_page():
    # background + styles (keep your existing CSS)
        # ---------- LOGIN PAGE BACKGROUND ----------
        # ---------- ORANGE → INDIGO GRADIENT BACKGROUND ----------
         # ---------- GLASS LOGIN CARD ----------
    st.markdown("""
<style>
/* 1️⃣ Disable all scrolling */
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    overflow: hidden !important;
}

/* 2️⃣ Streamlit app full viewport */
.stApp {
    height: 100vh;
    overflow: hidden !important;
}

/* 3️⃣ Remove Streamlit default padding */
section.main {
    padding: 0 !important;
}

/* 4️⃣ Hide scrollbar explicitly (extra safety) */
::-webkit-scrollbar {
    display: none;
}
</style>
""", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stForm {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }

        /* Inputs */
        .stTextInput input {
            background: rgba(255, 255, 255, 0.2);
            color: white;
           
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Labels */
        label {
            color: #fff3e0 !important;
            font-weight: 600;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(
                135deg,
                #7F00FF,
                #00C6FF
            ) !important;
            color: white !important;
            border-radius: 25px;
            font-weight: 700;
            padding: 10px;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(255, 152, 0, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
      
    st.markdown(
        """
        <style>
        html, body, .stApp {
            height: 100%;
            postion: fixed;
            
            
            overflow: hidden;
            margin: 0;
            padding: 0;

            /* Premium gradient */
            background: linear-gradient(
                180deg,
            
               
                #1e3c72 100%    /* Deep indigo bottom */
            );

            background-attachment: fixed;
            background-repeat: no-repeat;
        }

        header {
            display: none !important;
        }

        section.main {
            padding-top: 0rem;
        }
      
/* Remove default padding */
section.main {
    padding: 0 !important;
}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
<style>
/* Login form submit button gradient */
div[data-testid="stForm"] button {
     background: linear-gradient(135deg, #ff7a18, #ffb347) !important;
    color: blue !important;
    font-size: 44px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
    border-radius: 18px !important;
    border: none !important;
    transition: all 0.3s ease-in-out !important;
}

/* Hover effect */
div[data-testid="stForm"] button:hover {
    background: linear-gradient(135deg, #ffb347, #ff7a18) !important;
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(255, 122, 24, 0.8);
}
</style>
""", unsafe_allow_html=True)




    st.markdown("<h1 style='text-align:center;color:#00FFD1;'>🔐 Login </h1>", unsafe_allow_html=True)
    st.markdown("""
<style>
@keyframes shake {
  0% { transform: translateX(0); }
  20% { transform: translateX(-6px); }
  40% { transform: translateX(6px); }
  60% { transform: translateX(-6px); }
  80% { transform: translateX(6px); }
  100% { transform: translateX(0); }
}

@keyframes fadeOut {
  from { opacity: 1; }
  to { opacity: 0; }
}

.animated-error {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: 600;
    text-align: center;
    margin-top: 10px;
    animation: shake 0.4s ease, fadeOut 1s ease 7s forwards;
}
</style>
""", unsafe_allow_html=True)

    # ---- Toggle buttons ----
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):
            st.session_state.login_mode = "login"

    with col2:
        if st.button("Sign Up", use_container_width=True):
            st.session_state.login_mode = "signup"

    # ---- LOGIN FORM ----
    if st.session_state.login_mode == "login":
        with st.form("login_form"):
            username_email = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")

            if login_submit:
                if authenticate_user(username_email, password):
                    st.session_state.logged_in = True
                    st.session_state.page = "landing"
                    st.rerun()
                else:
                    error_box = st.empty()
                    error_box.markdown(
                        '<div class="animated-error">❌ Invalid username/email or password</div>',
                                   unsafe_allow_html=True)

                    time.sleep(8)          # ⏱ wait 8 seconds
                    error_box.empty();   # ❌ error hata do

    # ---- SIGNUP FORM ----
    elif st.session_state.login_mode == "signup":
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Create Account")

            if signup_submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    if register_user(new_username, new_email, new_password):
                        st.success("Account created 🎉")
                        st.session_state.login_mode = "login"
                        st.rerun()
                    else:
                        st.error("Username or email already exists ❌")



# ---------------- LANDING PAGE ----------------
# --- Fetch Crypto Data ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_crypto_data(symbols):
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        # Fetch data for the last 7 days + 2 days for 24h change calculation
        hist = ticker.history(period="8d") 
        
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            previous_close = hist["Close"].iloc[-2] if len(hist) >= 2 else current_price
            price_change = current_price - previous_close
            percentage_change = (price_change / previous_close) * 100 if previous_close != 0 else 0.0
            
            # Get last 7 days for sparkline, or fewer if not enough data
            sparkline_data = hist["Close"].tail(7).reset_index()
            sparkline_data['Date'] = sparkline_data['Date'].dt.date

            data[symbol] = {
                "current_price": current_price,
                "price_change": price_change,
                "percentage_change": percentage_change,
                "sparkline": sparkline_data
            }
        else:
            data[symbol] = {"current_price": None, "price_change": None, "percentage_change": None, "sparkline": pd.DataFrame()}
    return data

# --- Landing Page ---
def landing_page():
    set_page_background(gradient_start="#090D4AE1", gradient_end="#421DB8") # Dark gradient background

    col_buttons = st.columns(2)
    with col_buttons[0]:
        if st.button("Generate Price Forecast", use_container_width=True):
            st.session_state['page'] = 'dashboard'
            st.rerun()
    with col_buttons[1]:
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

    st.markdown(
        """
        <h1 style="
            text-align:center;
            color:#00FFD1;
            text-shadow: 2px 2px 10px black;
        ">
            Welcome to Crypto Price Predictor
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
<style>
/* General button styles for landing page */
div.stButton > button {
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 12px;
    border: none;
    transition: 0.3s ease;
    box-shadow: 0 0 15px rgba(255, 152, 0, 0.6); /* Default shadow */
}

/* Go to Dashboard button - first button in columns */
.stApp div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6a11cb, #2575fc); /* Blue-Purple gradient */
    box-shadow: 0 0 15px rgba(106, 17, 203, 0.6);
}
.stApp div[data-testid="stColumn"]:nth-of-type(1) div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #2575fc, #6a11cb);
    transform: scale(1.05);
}

/* Logout button - second button in columns */
.stApp div[data-testid="stColumn"]:nth-of-type(2) div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #ff5f6d, #ffc371); /* Red-Orange gradient */
    box-shadow: 0 0 15px rgba(255, 95, 109, 0.6);
}
.stApp div[data-testid="stColumn"]:nth-of-type(2) div[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #ffc371, #ff5f6d);
    transform: scale(1.05);
}

.market-card-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
   
    padding: 10px 0;
}

.market-card {
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    color:yellow;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 120px;
}
/* Total Market Trend */
.market-card.market-bearish h4 {
    color:purple !important;  /* Red */
}
.market-card.market-gainer h4 {
    color: #00FF99 !important;  /* Green */
}
.market-card.market-loser h4 {
    color:yellow;  /* Dark Red */
}
.market-card.market-neutral h4 {
    color:#00FF99; !important;  /* Blue */
}
               

.market-card h4 {
    margin-top: 0;
    margin-bottom: 5px;
    font-size: 1.1em;
    
}

.market-highlight {
    font-weight: bold;
    background-color: rgba(255, 255, 255, 0.2);
    padding: 3px 8px;
    border-radius: 5px;
    font-size: 1.2em;
    display: inline-block;
    margin-top: 5px;
}

.market-bearish {
    background: linear-gradient(135deg, #22c55e, #16a34e); /* Darker Red */
}
.market-gainer {
    background: linear-gradient(135deg, #388e3c, #1b5e20); /* Darker Green */
}

.market-loser {
    background: linear-gradient(135deg, #ff416c, #b31217); /* Darker Red */
}

    .market-neutral {
        background: linear-gradient(90deg, #1976d2, #0d47a1); /* Darker Blue */
    }
    .market-about-model {
        background: linear-gradient(135deg, #8e24aa, #4a148c); /* Purple gradient */
    }
    .footer-about-card {
        background: linear-gradient(135deg, #0082E0, #004D99); /* Blue gradient for About card */
    }
    .footer-feedback-card {
        background: linear-gradient(135deg, #FF6F00, #E65100); /* Orange gradient for Feedback card */
    }
</style>
""", unsafe_allow_html=True)
    # Fetch crypto data for the landing page
    crypto_symbols_map = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Ripple": "XRP-USD",
        "Dogecoin": "DOGE-USD",
        "Binancecoin": "BNB-USD"
    }
    symbols_list = list(crypto_symbols_map.values())
    crypto_data = get_crypto_data(symbols_list)
    

    # 1. Market Summary Section
    st.markdown("<h3 style=\"color:#1E90FF;\">📊 Market Summary</h3>", unsafe_allow_html=True)
    
    total_change_sum = 0
    valid_changes_count = 0
    top_gainer = {"name": "N/A", "change": -float('inf')}
    top_loser = {"name": "N/A", "change": float('inf')}

    for name, symbol in crypto_symbols_map.items():
        data = crypto_data.get(symbol, {})
        percentage_change = data.get("percentage_change")
        if percentage_change is not None:
            total_change_sum += percentage_change
            valid_changes_count += 1

            if percentage_change > top_gainer["change"]:
                top_gainer["name"] = name
                top_gainer["change"] = percentage_change
            if percentage_change < top_loser["change"]:
                top_loser["name"] = name
                top_loser["change"] = percentage_change

    avg_change = total_change_sum / valid_changes_count if valid_changes_count > 0 else 0

    market_trend = "Bullish 📈" if avg_change > 0 else "Bearish 📉" if avg_change < 0 else "Neutral ↔️"
    market_mood = "Greed" if avg_change > 1 else "Fear" if avg_change < -1 else "Neutral"

    st.markdown(f"""
    <div class="market-card-container">
        <div class="market-card {'market-bearish' if avg_change < 0 else 'market-gainer' if avg_change > 0 else 'market-neutral'}">
            <h4>Total Market Trend:</h4>
            <span class="market-highlight">{market_trend} ({avg_change:+.2f}%)</span>
        </div>
        <div class="market-card market-gainer">
            <h4>🔥 Top Gainer (24h):</h4>
            <span class="market-highlight">{top_gainer['name']} ({top_gainer['change']:+.2f}%)</span>
        </div>
        <div class="market-card market-loser">
            <h4>📉 Top Loser (24h):</h4>
            <span class="market-highlight">{top_loser['name']} ({top_loser['change']:+.2f}%)</span>
        </div>
        <div class="market-card market-neutral">
            <h4>💰 Market Mood:</h4>
            <span class="market-highlight">{market_mood}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    

    st.markdown("<h3 style=\"color:#1E90FF;\">Live Cryptocurrency Statistics</h3>", unsafe_allow_html=True)

    

    cols = st.columns(len(crypto_symbols_map))

    for i, (name, symbol) in enumerate(crypto_symbols_map.items()):
        with cols[i]:
            data = crypto_data.get(symbol, {})
            current_price = data.get("current_price")
            percentage_change = data.get("percentage_change")
            sparkline_df = data.get("sparkline", pd.DataFrame())

            if current_price is not None and percentage_change is not None:
                change_color_class = "positive-change" if percentage_change >= 0 else "negative-change"
                st.markdown(f"""
                <div class="crypto-card">
                    <div class="crypto-name">{name} ({symbol.split('-')[0]})</div>
                    <div class="crypto-price">${current_price:,.2f}</div>
                    <div class="price-change {change_color_class}">{percentage_change:+.2f}% (24h)</div>
                </div>
                """, unsafe_allow_html=True)
                if not sparkline_df.empty:
                    st.line_chart(sparkline_df.set_index('Date'))
                else:
                    st.markdown(f"""
                    <div class="crypto-card">
                        <div class="crypto-name">{name} ({symbol.split('-')[0]})</div>
                        <div class="crypto-price">N/A</div>
                        <div class="price-change">Data not available</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---<br>", unsafe_allow_html=True)

    

    # 3. Prediction Confidence Badge
    st.markdown("<h3 style=\"color:#1E90FF;\">🧠 Prediction Model Details</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="market-card-container">
        <div class="market-card market-neutral">
            <h4>Model Type:</h4>
            <span class="market-highlight">LSTM Neural Network</span>
        </div>
        <div class="market-card market-gainer">
            <h4>🎯 Accuracy (Train):</h4>
            <span class="market-highlight">~70%</span>
        </div>
        <div class="market-card market-neutral">
            <h4>⏱ Prediction Horizon:</h4>
            <span class="market-highlight">1–30 Days</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. About the Model Card
    st.markdown("<h3 style=\"color:#F97316;\">ℹ️ABOUT THE MODEL</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="market-card market-about-model">
        <p>This application utilizes Long Short-Term Memory (LSTM) neural networks, a type of recurrent neural network (RNN) well-suited for time series data like cryptocurrency prices. LSTMs are designed to remember long-term dependencies, making them effective at identifying patterns and trends in sequential data. Our model processes historical price data to predict future movements, offering insights into potential price trends. While sophisticated, it's important to remember that crypto markets are highly volatile and predictions are not guarantees.</p>
    </div>
    """, unsafe_allow_html=True)


    

    # New Footer Section
    st.markdown("""
    <div class="market-card-container" style="margin-top: 40px;">
        <div class="market-card footer-about-card">
            <h3>About Us</h3>
            <p>Contact us through:</p>
            <p><b>Gmail:</b> <a href="mailto:ynihal494@gmail.com" style="color: white;">cryptocurrency@gmail.com</a></p>
            <p><b>Contact Number:</b> 63781553XX</p>
            
      
            
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style=\"color:#FACC15;\">Give your valuable suggestion to improve this model:</h3>", unsafe_allow_html=True)
    with st.form(key='feedback_form_streamlit_footer', clear_on_submit=True):
        feedback_text_streamlit_footer = st.text_area("", placeholder="Enter your suggestions here...", label_visibility="collapsed", key='footer_feedback_text_area')
        submit_feedback_streamlit_footer = st.form_submit_button("Submit Feedback")
   

        if submit_feedback_streamlit_footer and feedback_text_streamlit_footer:
            st.success("Thank you for your precious suggestion! (Conceptually: Feedback received)")
        elif submit_feedback_streamlit_footer and not feedback_text_streamlit_footer:
            st.warning("Please enter your suggestion before submitting.")
    




# --- Main Dashboard Page (Original app logic) ---
def main_dashboard():
    set_page_background(image_file="Bitcoin-Price-Prediction-2023-735x400.png")

    if st.button("Go to Dashboard"):
        st.session_state['page'] = 'landing'
        st.rerun()
    st.markdown("""
<style>
/* Login button */
div.stButton > button {
    background: linear-gradient(135deg, #00E5FF, #FF5722);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 24px;
    border-radius: 12px;
    border: none;
    box-shadow: 0 0 15px rgba(255, 152, 0, 0.6);
    transition: 0.3s ease;
}

/* Hover effect */
div.stButton > button:hover {
    background: linear-gradient(135deg, #FF5722, #FF9800);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

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
    <p style="
        text-align:center;
        color:#FFD701;
        font-size:16px;
        font-weight:500;
        text-shadow: 1px 1px 6px black;
    ">
        Based on Yahoo Finance CRYPTO-USD data
    </p>
    """, unsafe_allow_html=True)


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


    prediction_days = 7


    show_prediction = True

    interval = st.selectbox(
        "Prediction Interval",
        ["1 Day", "7 Days", "30 Days"]
    )
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
    st.markdown("""
    <p style="
        text-align:center;
        color:#FFD701;
        font-size:16px;
        font-weight:500;
        text-shadow: 1px 1px 6px black;
    ">
        Based on Yahoo Finance CRYPTO-USD data
    </p>
    """, unsafe_allow_html=True)


    # ---------------- CURRENT PRICE ----------------
    ticker = yf.Ticker(selected_symbol)
    current_price = ticker.history(period="1d")["Close"].iloc[-1]

    st.markdown(f"""
    <div style="background:rgba(0,0,0,0.6);padding:20px;border-radius:15px;
    text-align:center;color:gold;font-size:28px;">
    💰 Current Price<br><b>${current_price:,.2f}</b>
    </div>
    """, unsafe_allow_html=True)
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


# --- Main Application Flow ---
# --- Main Application Flow ---
# --- Main Application Flow ---
# ---------------- APP ROUTING ----------------

if not st.session_state.logged_in:
    login_page()

else:
    if st.session_state.page == "landing":
        landing_page()

    elif st.session_state.page == "dashboard":
        main_dashboard()
