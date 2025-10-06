import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pytz
import feedparser

# -------------------------- PAGE CONFIG --------------------------
st.set_page_config(page_title="Indian Stock Monitor", page_icon="📈", layout="wide")

# -------------------------- DATABASE CONFIG --------------------------
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"
DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)
CACHE_TTL_MIN = 5

# -------------------------- DATA LOADERS --------------------------
@st.cache_data(ttl=CACHE_TTL_MIN*60)
def load_table(table_name):
    cache_file = f"{DATA_PATH}/{table_name}.csv"
    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        df.to_csv(cache_file, index=False)
        return df
    except:
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
    return pd.DataFrame()

stocks_df = load_table("stocks")
news_df = load_table("news_sentiment")
pred_df = load_table("buy_sell_predictions")

# -------------------------- MARKET INFO --------------------------
IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()
market_open = "09:15 AM"
market_close = "03:30 PM"

# Weekend adjustment
if today.weekday() == 5: last_trading_day = today - timedelta(days=1)
elif today.weekday() == 6: last_trading_day = today - timedelta(days=2)
else: last_trading_day = today

last_updated = now_ist.strftime("%Y-%m-%d %H:%M:%S %Z")
st.markdown(
    f"""
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                background-color:#fff3cd; padding:10px 20px; border-radius:8px; 
                border:1px solid #ffeeba; font-size:14px; z-index:9999;">
        Market Open: {market_open} | Market Close: {market_close} | Last Updated: {last_updated}
    </div>
    """, unsafe_allow_html=True
)

# -------------------------- SIDEBAR --------------------------
st.sidebar.header("⚙️ Stock & Dashboard Controls")
all_symbols = sorted(stocks_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks", all_symbols, default=all_symbols[:3])
granularity = st.sidebar.radio("Candlestick Timeframe", ["Daily", "Monthly", "Yearly"])
refresh = st.sidebar.button("🔄 Refresh Data")
if refresh:
    st.cache_data.clear()
    st.rerun()

# -------------------------- ALERT NOTIFICATION --------------------------
def get_top_alerts():
    alerts = []
    if not stocks_df.empty:
        latest = stocks_df.groupby("symbol").last().reset_index()
        latest["change"] = latest["close"] - latest["open"]
        top_gainers = latest.sort_values("change", ascending=False).head(2)
        top_losers = latest.sort_values("change").head(2)
        for _, row in top_gainers.iterrows():
            alerts.append(f"📈 Gainer: {row['symbol']} ({row['change']:.2f})")
        for _, row in top_losers.iterrows():
            alerts.append(f"📉 Loser: {row['symbol']} ({row['change']:.2f})")
    if not pred_df.empty:
        top_buy = pred_df[pred_df['action']=='buy'].head(2)
        top_sell = pred_df[pred_df['action']=='sell'].head(2)
        for _, row in top_buy.iterrows():
            alerts.append(f"⬆️ Buy: {row['symbol']} ({row['buy_pred']*100:.1f}%)")
        for _, row in top_sell.iterrows():
            alerts.append(f"⬇️ Sell: {row['symbol']} ({row['sell_pred']*100:.1f}%)")
    if not news_df.empty:
        recent_news = news_df.head(2)
        for _, row in recent_news.iterrows():
            alerts.append(f"📰 News: {row['title'][:50]}...")
    return alerts

alerts = get_top_alerts()
if alerts:
    alert_text = " | ".join(alerts)
    st.markdown(f"<div style='background-color:#fff3cd; padding:10px; border-radius:8px;'>{alert_text}</div>", unsafe_allow_html=True)

# -------------------------- DASHBOARD LAYOUT --------------------------
col_main, col_chat = st.columns([3,1])  # 75% main, 25% AI chatbot

with col_main:
    st.title("📊 Indian Stock Dashboard")

    # ---- PRICE TREND LINE CHART ----
    st.subheader("📈 Stock Closing Prices")
    if not stocks_df.empty and selected_symbols:
        fig = px.line(stocks_df[stocks_df["symbol"].isin(selected_symbols)],
                      x="timestamp", y="close", color="symbol",
                      title="Stock Closing Prices")
        st.plotly_chart(fig, use_container_width=True)

    # ---- CANDLESTICK CHART ----
    st.subheader("📊 Candlestick Chart")
    for symbol in selected_symbols:
        df_symbol = stocks_df[stocks_df["symbol"]==symbol].copy()
        df_symbol["timestamp"] = pd.to_datetime(df_symbol["timestamp"])
        if granularity=="Monthly":
            df_symbol = df_symbol.resample('M', on='timestamp').agg({'open':'first','high':'max','low':'min','close':'last'}).reset_index()
        elif granularity=="Yearly":
            df_symbol = df_symbol.resample('Y', on='timestamp').agg({'open':'first','high':'max','low':'min','close':'last'}).reset_index()
        fig = go.Figure(data=[go.Candlestick(
            x=df_symbol['timestamp'], open=df_symbol['open'], high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close'], name=symbol)])
        fig.update_layout(title=f"{symbol} Candlestick Chart ({granularity})", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # ---- BUY/SELL PREDICTIONS TABLE ----
    st.subheader("💹 Buy/Sell Recommendations")
    if not pred_df.empty:
        display_df = pred_df.copy()
        display_df["Action"] = display_df["action"].map({"buy":"⬆️ Buy","sell":"⬇️ Sell","no trade":"⏸️ No Trade"})
        st.dataframe(display_df[["symbol","Action","buy_pred","sell_pred"]].head(5), use_container_width=True)

    # ---- NEWS HEADLINES ----
    st.subheader("📰 Latest News & Sentiment")
    if not news_df.empty:
        news_display = news_df[["symbol","title","sentiment"]].head(5).copy()
        st.dataframe(news_display, use_container_width=True)

# -------------------------- GEN AI CHATBOT --------------------------
with col_chat:
    st.header("💬 Financial AI Assistant")
    user_query = st.text_area("Ask a question about selected stocks:")
    if st.button("🤖 Ask AI"):
        if user_query.strip():
            # Simulate AI response (replace with GPT API for real)
            ai_response = f"Analyzing query: '{user_query}' for stocks {selected_symbols}...\n"
            ai_response += "Recent trends: bullish for selected symbols. Check buy/sell predictions and news sentiment."
            st.markdown(ai_response)
