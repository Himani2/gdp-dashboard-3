import os
import pandas as pd
import streamlit as st
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================
# DATABASE CONFIGURATION
# ================================
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"


DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(DB_URI)

# ================================
# LOAD DATA FUNCTION
# ================================
@st.cache_data(ttl=300)
def load_table(table_name):
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to load {table_name}: {e}")
        return pd.DataFrame()

# ================================
# FETCH DATA
# ================================
stocks_df = load_table("stocks")
pred_df = load_table("buy_sell_recommendation")
news_df = load_table("news_sentiment")
earnings_df = pd.read_csv("earnings_calendar.csv") if os.path.exists("earnings_calendar.csv") else pd.DataFrame()

# ================================
# TIME SETTINGS
# ================================
IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()

# ================================
# SIDEBAR - STOCK SELECTION & WATCHLIST
# ================================
st.sidebar.title("📊 Dashboard Controls")
all_symbols = sorted(stocks_df["symbol"].unique()) if not stocks_df.empty else []
selected_symbols = st.sidebar.multiselect("Select Stocks for Candlestick Chart", all_symbols, default=all_symbols[:3])
watchlist_symbols = st.sidebar.multiselect("Select Watchlist Stocks", all_symbols, default=all_symbols[:3])

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# ================================
# ALERTS (Top Confident Trades)
# ================================
ALERTS_CSV = "alerts_history.csv"
alerts = []

if not pred_df.empty:
    for _, row in pred_df.iterrows():
        alerts.append({
            "symbol": row["symbol"],
            "action": row["action"],
            "confidence": f"{row['buy_pred']*100:.1f}%" if row["action"]=="BUY" else f"{row['sell_pred']*100:.1f}%",
            "timestamp": row["timestamp"]
        })

# Save alerts to CSV
if alerts:
    alert_df = pd.DataFrame(alerts)
    if os.path.exists(ALERTS_CSV):
        alert_df.to_csv(ALERTS_CSV, mode='a', header=False, index=False)
    else:
        alert_df.to_csv(ALERTS_CSV, index=False)

st.subheader(f"🔔 Alerts ({len(alerts)})")
if alerts:
    st.dataframe(pd.DataFrame(alerts)[["symbol","action","confidence","timestamp"]], hide_index=True, use_container_width=True)
else:
    st.info("No active alerts.")

# ================================
# OPEN / CLOSE / TOTAL TRADES METRICS
# ================================
st.subheader("📊 Open / Close Trade Metrics (ML Predictions)")
if not pred_df.empty:
    open_conf = (pred_df[pred_df["action"]=="BUY"]["buy_pred"].sum()*100).round(1)
    close_conf = (pred_df[pred_df["action"]=="SELL"]["sell_pred"].sum()*100).round(1)
    total_trades = pred_df.shape[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 Open Trade Confidence", f"{open_conf:.1f}%")
    m2.metric("🔴 Close Trade Confidence", f"{close_conf:.1f}%")
    m3.metric("💼 Total Trades", f"{total_trades}")

# ================================
# MARKET INDICES METRIC CARDS
# ================================
st.subheader("📈 Market Indices")
indices_symbols = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY 500": "^CRSLDX"
}
indices_data = []
for name,symbol in indices_symbols.items():
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        last_close = hist["Close"].iloc[-1]
        prev_close = hist["Close"].iloc[-2]
        pct_change = ((last_close - prev_close)/prev_close)*100
        arrow = "⬆️" if pct_change>=0 else "⬇️"
        indices_data.append({"name":name,"price":last_close,"change":f"{arrow} {pct_change:.2f}%"})
    except:
        indices_data.append({"name":name,"price":0,"change":"N/A"})

cols = st.columns(len(indices_data))
for i,row in enumerate(indices_data):
    cols[i].metric(label=row["name"], value=f"₹ {row['price']:.2f}", delta=row["change"])

# ================================
# TOP GAINERS / LOSERS / MOST / LEAST ACTIVE
# ================================
st.subheader("🏆 Market Movers")
if not stocks_df.empty:
    latest = stocks_df.groupby("symbol").last().reset_index()
    latest["change_pct"] = ((latest["close"] - latest["open"])/latest["open"])*100
    top_gainers = latest.nlargest(5,"change_pct")[["symbol","close","change_pct"]]
    top_losers = latest.nsmallest(5,"change_pct")[["symbol","close","change_pct"]]
    most_active = latest.nlargest(5,"volume")[["symbol","close","volume"]]
    least_active = latest.nsmallest(5,"volume")[["symbol","close","volume"]]

    c1,c2,c3,c4 = st.columns(4)
    def format_arrow(val):
        return f"🟢 {val:.2f}%" if val>0 else f"🔴 {val:.2f}%"
    with c1:
        st.markdown("**🟢 Top Gainers**")
        top_gainers["change_pct"] = top_gainers["change_pct"].apply(format_arrow)
        st.dataframe(top_gainers, hide_index=True)
    with c2:
        st.markdown("**🔴 Top Losers**")
        top_losers["change_pct"] = top_losers["change_pct"].apply(format_arrow)
        st.dataframe(top_losers, hide_index=True)
    with c3:
        st.markdown("**💼 Most Active**")
        st.dataframe(most_active, hide_index=True)
    with c4:
        st.markdown("**📉 Least Active**")
        st.dataframe(least_active, hide_index=True)

# ================================
# WATCHLIST
# ================================
if watchlist_symbols:
    st.subheader("⭐ Watchlist Metrics")
    watch_df = stocks_df[stocks_df["symbol"].isin(watchlist_symbols)].groupby("symbol").last().reset_index()
    cols = st.columns(len(watch_df))
    for i,row in watch_df.iterrows():
        delta = row["close"]-row["open"]
        arrow = "⬆️" if delta>0 else "⬇️"
        cols[i].metric(label=row["symbol"], value=f"₹ {row['close']:.2f}", delta=f"{arrow} {delta:.2f}")

# ================================
# CANDLESTICK CHART
# ================================
if selected_symbols:
    st.subheader("📊 Candlestick Chart")
    for sym in selected_symbols:
        df = stocks_df[stocks_df["symbol"]==sym].sort_values("timestamp")
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )])
        fig.update_layout(title=f"{sym} Candlestick", xaxis_title="Time", yaxis_title="Price", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# ================================
# BUY / SELL RECOMMENDATIONS
# ================================
st.subheader("💹 ML Trade Recommendations")
if not pred_df.empty:
    pred_df["Buy %"] = (pred_df["buy_pred"]*100).round(1)
    pred_df["Sell %"] = (pred_df["sell_pred"]*100).round(1)
    st.dataframe(pred_df[["symbol","action","Buy %","Sell %","timestamp"]], hide_index=True, use_container_width=True)

# ================================
# NEWS (Sidebar Scrollable)
# ================================
st.sidebar.title("📰 Latest News")
if not news_df.empty:
    if "news_idx" not in st.session_state:
        st.session_state.news_idx = 0
    def prev_news(): st.session_state.news_idx = max(0, st.session_state.news_idx-1)
    def next_news(): st.session_state.news_idx = min(len(news_df)-1, st.session_state.news_idx+1)
    ncol1,ncol2,ncol3 = st.sidebar.columns([1,6,1])
    with ncol1: st.button("⬅️", on_click=prev_news)
    with ncol3: st.button("➡️", on_click=next_news)
    news_row = news_df.iloc[st.session_state.news_idx]
    sentiment_icon = "🟢" if news_row["sentiment"].lower() in ["positive","bullish"] else "🔴" if news_row["sentiment"].lower() in ["negative","bearish"] else "⚪"
    st.sidebar.markdown(f"**{sentiment_icon} {news_row['title']}**  \n*Source:* {news_row.get('source','N/A')}")

# ================================
# EARNINGS CALENDAR (Sidebar)
# ================================
st.sidebar.title("📅 Earnings / Events Calendar")
events_per_page = 1
if "calendar_page" not in st.session_state:
    st.session_state.calendar_page = 0

total_pages = (len(earnings_df)-1)//events_per_page+1 if not earnings_df.empty else 1
def prev_event(): st.session_state.calendar_page = max(0, st.session_state.calendar_page-1)
def next_event(): st.session_state.calendar_page = min(total_pages-1, st.session_state.calendar_page+1)

c1,c2,c3 = st.sidebar.columns([1,6,1])
with c1: st.button("⬅️", on_click=prev_event)
with c3: st.button("➡️", on_click=next_event)

if not earnings_df.empty:
    start_idx = st.session_state.calendar_page*events_per_page
    page_events = earnings_df.iloc[start_idx:start_idx+events_per_page]
    for _,row in page_events.iterrows():
        st.sidebar.markdown(f"**{row['symbol']}**  \n{row['date']}  \n*Event:* {row.get('event','N/A')}")
else:
    st.sidebar.info("No earnings/events found.")

st.success("✅ Dashboard Loaded Successfully")
