import os
import pandas as pd
import streamlit as st
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import plotly.express as px

# =======================================
# DATABASE CONFIGURATION
# =======================================


DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(DB_URI)

# =======================================
# LOAD DATA FUNCTION
# =======================================
@st.cache_data(ttl=300)
def load_table(table_name):
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to load {table_name}: {e}")
        return pd.DataFrame()

# =======================================
# FETCH DATA
# =======================================
stocks_df = load_table("stocks")
pred_df = load_table("buy_sell_recommendation")
news_df = load_table("news_sentiment")

# =======================================
# TIME SETTINGS
# =======================================
IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()

# =======================================
# SIDEBAR - STOCK SELECTION
# =======================================
st.sidebar.title("📊 Stock Dashboard Controls")

all_symbols = sorted(stocks_df["symbol"].unique()) if not stocks_df.empty else []
selected_symbols = st.sidebar.multiselect("Select Stocks for Chart", all_symbols, default=all_symbols[:3])

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# =======================================
# MARKET INDICES (Yahoo Finance)
# =======================================
st.title("📈 Indian Market Overview")

indices_symbols = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY 500": "^CRSLDX"
}

indices_data = []
for name, symbol in indices_symbols.items():
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d")["Close"].iloc[-1]
        indices_data.append({"name": name, "price": price})
    except:
        pass

cols = st.columns(len(indices_data))
for i, row in enumerate(indices_data):
    cols[i].metric(label=row["name"], value=f"₹ {row['price']:.2f}")

# =======================================
# ALERT NOTIFICATION (Bell Icon)
# =======================================
ALERTS_CSV = "alerts_history.csv"
alerts = []

if not pred_df.empty:
    for _, row in pred_df.iterrows():
        alerts.append({
            "symbol": row["symbol"],
            "action": row["action"],
            "confidence": f"{row['confidence']*100:.1f}%",
            "timestamp": row["timestamp"]
        })

# Save alerts to CSV
if alerts:
    alert_df = pd.DataFrame(alerts)
    if os.path.exists(ALERTS_CSV):
        alert_df.to_csv(ALERTS_CSV, mode='a', header=False, index=False)
    else:
        alert_df.to_csv(ALERTS_CSV, index=False)

with st.container():
    st.markdown(f"### 🔔 Notifications ({len(alerts)})")
    if st.button("View Notifications"):
        st.dataframe(pd.DataFrame(alerts)[["symbol", "action", "confidence", "timestamp"]],
                     use_container_width=True, hide_index=True)

# =======================================
# MARKET SENTIMENT BAR
# =======================================
if not pred_df.empty:
    buy_conf = (pred_df[pred_df["action"] == "BUY"]["confidence"].mean()) * 100
    sell_conf = (pred_df[pred_df["action"] == "SELL"]["confidence"].mean()) * 100
    market_sentiment = "🟢 Bullish" if buy_conf > sell_conf else "🔴 Bearish"
    st.metric("Market Sentiment", market_sentiment, f"Buy: {buy_conf:.1f}% | Sell: {sell_conf:.1f}%")

# =======================================
# TOP GAINERS / LOSERS / ACTIVE STOCKS
# =======================================
st.subheader("🏆 Market Movers")

if not stocks_df.empty:
    latest_df = stocks_df.groupby("symbol").last().reset_index()
    latest_df["change_pct"] = ((latest_df["close"] - latest_df["open"]) / latest_df["open"]) * 100

    gainers = latest_df.nlargest(5, "change_pct")[["symbol", "close", "change_pct"]]
    losers = latest_df.nsmallest(5, "change_pct")[["symbol", "close", "change_pct"]]
    active = latest_df.nlargest(5, "volume")[["symbol", "close", "volume"]]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🟢 Top Gainers**")
        st.dataframe(gainers, hide_index=True)
    with c2:
        st.markdown("**🔴 Top Losers**")
        st.dataframe(losers, hide_index=True)
    with c3:
        st.markdown("**💼 Most Active**")
        st.dataframe(active, hide_index=True)

# =======================================
# STOCK PRICE CHART
# =======================================
if not stocks_df.empty and selected_symbols:
    st.subheader("📊 Stock Price Trends")
    df_chart = stocks_df[stocks_df["symbol"].isin(selected_symbols)]
    fig = px.line(df_chart, x="timestamp", y="close", color="symbol", title="Price Movement")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# =======================================
# BUY/SELL RECOMMENDATIONS
# =======================================
st.subheader("💹 ML Model Trade Recommendations")
if not pred_df.empty:
    selected_action = st.selectbox("Filter by Action", ["All", "BUY", "SELL"])
    display_df = pred_df.copy()
    display_df["Buy %"] = (display_df["buy_pred"]*100).round(1)
    display_df["Sell %"] = (display_df["sell_pred"]*100).round(1)

    if selected_action != "All":
        display_df = display_df[display_df["action"] == selected_action]

    st.dataframe(display_df[["symbol", "action", "Buy %", "Sell %", "timestamp"]],
                 hide_index=True, use_container_width=True)

# =======================================
# EARNINGS CALENDAR (Yahoo Finance)
# =======================================
@st.cache_data(ttl=3600)
def get_earnings_calendar():
    symbols = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "SBIN.NS", "ICICIBANK.NS", "WIPRO.NS"]
    all_events = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            earnings = ticker.get_earnings_dates(limit=20)
            if not earnings.empty:
                earnings = earnings.reset_index().rename(columns={"Earnings Date": "date"})
                earnings["symbol"] = sym
                all_events.append(earnings)
        except Exception as e:
            print(f"Error fetching earnings for {sym}: {e}")
    if all_events:
        df = pd.concat(all_events)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= "2026-06-30"]
        df = df.sort_values("date")
        return df
    return pd.DataFrame()

earnings_df = get_earnings_calendar()

# --- Sidebar Earnings Calendar ---
st.sidebar.title("📅 Quarterly Earnings Calendar")
events_per_page = 5
if "calendar_page" not in st.session_state:
    st.session_state.calendar_page = 0

total_pages = (len(earnings_df) - 1) // events_per_page + 1 if not earnings_df.empty else 1

def prev_page():
    if st.session_state.calendar_page > 0:
        st.session_state.calendar_page -= 1

def next_page():
    if st.session_state.calendar_page < total_pages - 1:
        st.session_state.calendar_page += 1

col1, col2, col3 = st.sidebar.columns([1,6,1])
with col1:
    st.button("⬅️ Prev", on_click=prev_page)
with col3:
    st.button("Next ➡️", on_click=next_page)

if not earnings_df.empty:
    start_idx = st.session_state.calendar_page * events_per_page
    end_idx = start_idx + events_per_page
    page_events = earnings_df.iloc[start_idx:end_idx]

    for _, row in page_events.iterrows():
        date_str = row['date'].strftime("%b %d, %Y")
        st.sidebar.metric(label=row["symbol"], value=date_str)
else:
    st.sidebar.info("No earnings data found.")

# =======================================
# NEWS & EVENTS IN CARD STYLE
# =======================================
st.subheader("🗞️ Latest Market News")
if not news_df.empty:
    for _, row in news_df.head(5).iterrows():
        with st.container():
            sentiment_icon = "🟢" if row["sentiment"].lower() in ["positive", "bullish"] else "🔴" if row["sentiment"].lower() in ["negative", "bearish"] else "⚪"
            st.markdown(f"""
            **{sentiment_icon} {row['title']}**  
            *Source:* {row.get('source', 'N/A')}  
            """)
else:
    st.info("No news available currently.")

st.success("✅ Dashboard Loaded Successfully")
