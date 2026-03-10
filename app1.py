import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import ta
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Quant Financial Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# AUTO REFRESH
# ---------------------------------------------------

st_autorefresh(interval=60000, key="refresh")

# ---------------------------------------------------
# LANDING PAGE
# ---------------------------------------------------

if "start" not in st.session_state:
    st.session_state.start = False

if not st.session_state.start:

    st.title("🚀 AI Quant Financial Analytics Platform")

    st.markdown("""
    ### Advanced Stock Intelligence Dashboard

    ✔ AI Forecasting  
    ✔ Technical Indicators  
    ✔ Portfolio Optimizer  
    ✔ Risk Analyzer  
    ✔ Global Markets  
    ✔ Crypto Tracker
    """)

    if st.button("Enter Dashboard"):
        st.session_state.start = True

    st.stop()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙ Dashboard Settings")

ticker = st.sidebar.text_input("Stock Ticker", "TCS.NS")

period = st.sidebar.selectbox(
    "Select Period",
    ["3mo","6mo","1y","2y","5y"]
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data(ticker, period):

    try:
        df = yf.download(ticker, period=period, progress=False)

        if df is None or df.empty:
            return None

        df.reset_index(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"])

        return df

    except Exception:
        return None
    


data = load_data(ticker, period)

if data is None or data.empty:
    st.error("⚠ Unable to fetch stock data. Please check the ticker symbol.")
    st.stop()

# Fix yfinance MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


# ---------------------------------------------------
# INDICATORS
# ---------------------------------------------------

# Make sure Close is numeric
data["Close"] = pd.to_numeric(data["Close"].squeeze(), errors="coerce")
data.dropna(inplace=True)

close = data["Close"]

# Calculate indicators
data["SMA"] = ta.trend.sma_indicator(close=close, window=20)

data["EMA"] = ta.trend.ema_indicator(close=close, window=20)

data["RSI"] = ta.momentum.rsi(close=close, window=14)

data["MACD"] = ta.trend.macd(close=close)

data["MACD_SIGNAL"] = ta.trend.macd_signal(close=close)

# Remove NaN rows created by indicators
data.dropna(inplace=True)
# ---------------------------------------------------
# SIGNAL
# ---------------------------------------------------

if len(data) > 0:
    rsi = data["RSI"].iloc[-1]
else:
    rsi = 50

if rsi > 70:
    signal = "SELL 🔴"
elif rsi < 30:
    signal = "BUY 🟢"
else:
    signal = "HOLD 🟡"

# ---------------------------------------------------
# STOCK INFO
# ---------------------------------------------------

try:
    stock = yf.Ticker(ticker)
    info = stock.fast_info

    price = round(info.get("lastPrice",0),2)
    high = round(info.get("dayHigh",0),2)
    low = round(info.get("dayLow",0),2)
    market_cap = info.get("marketCap",0)

except:
    price, high, low, market_cap = 0,0,0,0

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.title("📊 AI Quant Financial Dashboard")

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------

c1,c2,c3,c4,c5 = st.columns(5)

c1.metric("💰 Price", price)
c2.metric("📊 Market Cap", f"{market_cap:,}")
c3.metric("📈 Day High", high)
c4.metric("📉 Day Low", low)
c5.metric("🤖 AI Signal", signal)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
"Market",
"Indicators",
"AI Forecast",
"Portfolio",
"Risk",
"Global Markets"
])

# ---------------------------------------------------
# MARKET
# ---------------------------------------------------

with tab1:

    st.subheader("📊 Candlestick Chart")

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data["Date"],
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 Volume")

    vol = go.Figure()

    vol.add_trace(go.Bar(
        x=data["Date"],
        y=data["Volume"],
        name="Volume"
    ))

    vol.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume"
    )

    st.plotly_chart(vol, use_container_width=True)

# ---------------------------------------------------
# INDICATORS
# ---------------------------------------------------

with tab2:

    st.subheader("Moving Averages")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], name="SMA"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["EMA"], name="EMA"))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RSI")

    rsi_fig = go.Figure()

    rsi_fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI"))

    rsi_fig.add_hline(y=70)
    rsi_fig.add_hline(y=30)

    st.plotly_chart(rsi_fig, use_container_width=True)

    st.subheader("MACD")

    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD_SIGNAL"], name="Signal"))

    st.plotly_chart(macd_fig, use_container_width=True)

# ---------------------------------------------------
# AI FORECAST
# ---------------------------------------------------

with tab3:

    st.subheader("🤖 AI Price Forecast")

    df_prophet = pd.DataFrame()

    df_prophet["ds"] = pd.to_datetime(data["Date"])
    df_prophet["y"] = data["Close"].astype(float)

    df_prophet = df_prophet.dropna()

    if len(df_prophet) < 10:

        st.warning("Not enough data for forecast. Try longer period.")

    else:

        model = Prophet()

        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30)

        forecast = model.predict(future)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_prophet["ds"],
            y=df_prophet["y"],
            name="Actual"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Forecast"
        ))

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# PORTFOLIO
# ---------------------------------------------------

with tab4:

    st.subheader("Portfolio Optimizer")

    stocks = st.text_input("Enter Stocks","TCS.NS,INFY.NS,HDFCBANK.NS")

    tickers = stocks.split(",")

    prices = yf.download(tickers, period="1y")["Close"]

    returns = prices.pct_change().dropna()

    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    port_return = np.sum(returns.mean()*weights)*252
    port_vol = np.sqrt(np.dot(weights.T,np.dot(returns.cov()*252,weights)))

    sharpe = port_return/port_vol

    st.write("Expected Return:", round(port_return,3))
    st.write("Volatility:", round(port_vol,3))
    st.write("Sharpe Ratio:", round(sharpe,3))

# ---------------------------------------------------
# RISK
# ---------------------------------------------------

with tab5:

    st.subheader("Value at Risk")

    returns = data["Close"].pct_change().dropna()

    var = np.percentile(returns,5)

    st.write("Daily VaR:", round(var,4))

# ---------------------------------------------------
# GLOBAL MARKETS
# ---------------------------------------------------

with tab6:

    st.subheader("Global Markets")

    markets = {
        "S&P 500":"^GSPC",
        "NASDAQ":"^IXIC",
        "DOW JONES":"^DJI",
        "NIFTY 50":"^NSEI"
    }

    rows=[]

    for name,symbol in markets.items():

        try:
            price = yf.Ticker(symbol).fast_info["lastPrice"]
            rows.append([name,price])
        except:
            rows.append([name,"N/A"])

    df = pd.DataFrame(rows,columns=["Market","Price"])

    st.dataframe(df)

    st.subheader("Crypto")

    crypto = {
        "Bitcoin":"BTC-USD",
        "Ethereum":"ETH-USD",
        "Solana":"SOL-USD"
    }

    rows=[]

    for name,symbol in crypto.items():

        try:
            price = yf.Ticker(symbol).fast_info["lastPrice"]
            rows.append([name,price])
        except:
            rows.append([name,"N/A"])

    df = pd.DataFrame(rows,columns=["Crypto","Price"])

    st.dataframe(df)

# ---------------------------------------------------
# EXPORT
# ---------------------------------------------------

st.subheader("Export Data")

csv = data.to_csv(index=False).encode()

st.download_button(
    "Download CSV",
    csv,
    file_name=f"{ticker}_data.csv",
    mime="text/csv"
)