import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import requests
from datetime import datetime
import time
from bs4 import BeautifulSoup
import pytz
import feedparser
# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Indian Stock Monitor", page_icon="📈", layout="wide")


# --------------------------
# DATABASE CONFIG
# --------------------------
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"

DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)
CACHE_TTL = "4h"

# --------------------------
# DATA LOADER
# --------------------------
@st.cache_data(ttl=CACHE_TTL)
def load_or_fetch(table_name: str):
    cache_file = f"{DATA_PATH}/{table_name}.csv"
    df = None
    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        st.success(f"✅ Loaded {table_name} from Database")
    except Exception as e:
        st.warning(f"⚠️ Database load failed for {table_name}: {e}")

    if df is not None and not df.empty:
        if table_name != "stocks":
            df.to_csv(cache_file, index=False)
        return df

    if os.path.exists(cache_file):
        st.info(f"📁 Using cached {table_name}.csv")
        return pd.read_csv(cache_file)

    st.error(f"❌ No data available for {table_name}")
    return pd.DataFrame()

    

# --------------------------
# LOAD TABLES
# --------------------------
stocks_df = load_or_fetch("stocks")
news_df = load_or_fetch("news_sentiment")
pred_df = load_or_fetch("buy_sell_predictions")

# # --------------------------
# # MARKET DATE LOGIC
# # --------------------------
# today = datetime.now().date()
# if today.weekday() >= 5:  # Sat/Sun
#     last_friday = today - timedelta(days=today.weekday() - 4)
#     st.warning(f"Market closed 🛑 Showing last Friday ({last_friday}) data")
#     stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])
#     stocks_df = stocks_df[stocks_df["timestamp"] <= pd.Timestamp(last_friday)]


# --------------------------
# MARKET DATE LOGIC
# --------------------------
# --------------------------
# MARKET INFO LOGIC
# --------------------------
IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()

# Market open/close times (IST)
market_open = "09:15 AM"
market_close = "03:30 PM"

# Determine last trading day for weekend
if today.weekday() == 5:  # Saturday
    last_trading_day = today - timedelta(days=1)
    weekend_warning = f"Market closed 🛑 Showing last Friday ({last_trading_day}) data"
elif today.weekday() == 6:  # Sunday
    last_trading_day = today - timedelta(days=2)
    weekend_warning = f"Market closed 🛑 Showing last Friday ({last_trading_day}) data"
else:
    last_trading_day = today
    weekend_warning = None

# Last updated timestamp in IST
last_updated = now_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

# --------------------------
# Display in top-right corner
# --------------------------
warning_html = f"<br><span style='color:red;'>{weekend_warning}</span>" if weekend_warning else ""

st.markdown(
    f"""
    <div style="
        position: fixed;
        top: 10px;
        right: 10px;
        background-color:#f0f2f6;
        padding:10px 15px;
        border-radius:8px;
        box-shadow:0 2px 5px rgba(0,0,0,0.1);
        font-size:14px;
        color:#333;
        z-index:9999;
        text-align:right;
    ">
        <b>Market Open:</b> {market_open}  <br>
        <b>Market Close:</b> {market_close} <br>
        <b>Last Updated:</b> {last_updated}
        {warning_html}
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------
# SIDEBAR CONTROLS
# --------------------------
st.sidebar.header("⚙️ Select Indian Stocks you would like to filter")
all_symbols = sorted(stocks_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks", all_symbols, default=all_symbols[:3])
refresh = st.sidebar.button("🔄 Refresh Data")

if refresh:
    st.cache_data.clear()
    st.rerun()

# --------------------------
# MAIN + CHAT LAYOUT
# --------------------------
col_main, col_chat = st.columns([3, 1])  # 75% main, 25% chat
# --------------------------
# MAIN DASHBOARD
# --------------------------
with col_main:
    st.title("📊 Indian Stock Dashboard")

# # ---------------BUY/SELL RECOMMENDER---------------------------------------------------------------

st.subheader("📊 Buy/Sell Recommender: ")
# --- Dropdown for stock selection ---
all_symbols = sorted(pred_df["symbol"].unique())
selected_stock = st.selectbox("Select a Stock", all_symbols)

# --- Fetch prediction for selected stock ---
if selected_stock:
    rec = pred_df[pred_df["symbol"] == selected_stock]
    if not rec.empty:
        buy_pred = rec.iloc[0]["buy_pred"]
        sell_pred = rec.iloc[0]["sell_pred"]
        action = rec.iloc[0]["action"]

        # --- Display recommendation ---
        st.success(
            f"✅ **{selected_stock}** → Model suggests **{action}**\n\n"
            f"**Buy Confidence:** {buy_pred*100:.1f}%  \n"
            f"**Sell Confidence:** {sell_pred*100:.1f}%"
        )
    else:
        st.info(f"No prediction available for {selected_stock}.")

  #-----------------------------------------------------------------
# --- File paths for cache ---
INDEX_CSV = "indices_cache.csv"
RSS_CSV = "economic_news_cache.csv"
CACHE_TTL_MIN = 5  # minutes

# --- Define Indian Indices ---
st.subheader("📊 Indian Market Indices ")

indices = {
    "NIFTY 50": "NSEI",
    "SENSEX": "BSESN",
    "NIFTY BANK": "NSEBANK",
    "NIFTY 500": "CRSLDX"
}

# --- Google Finance fetch ---
def fetch_google_finance(symbol):
    try:
        url = f"https://www.google.com/finance/quote/{symbol}:NSE"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        price_tag = soup.find("div", class_="YMlKec fxKbKc")
        if price_tag:
            return float(price_tag.text.replace(",", ""))
    except:
        return None
    return None

# --- Economic RSS fetch ---
RSS_URL = "https://news.google.com/rss/search?q=India+economy&hl=en-IN&gl=IN&ceid=IN:en"

def fetch_economic_rss():
    feed = feedparser.parse(RSS_URL)
    news_items = []
    for entry in feed.entries[:5]:  # latest 5 news
        news_items.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })
    df = pd.DataFrame(news_items)
    df["published"] = pd.to_datetime(df["published"])
    return df

# --- Helper to check if cache is fresh ---
def is_cache_fresh(file_path, ttl_minutes=CACHE_TTL_MIN):
    if os.path.exists(file_path):
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if datetime.now() - file_time < timedelta(minutes=ttl_minutes):
            return True
    return False

# --- Load indices with CSV cache ---
def load_index_data():
    if is_cache_fresh(INDEX_CSV):
        df = pd.read_csv(INDEX_CSV)
        df["price"] = df["price"].astype(float)
    else:
        data = []
        for name, symbol in indices.items():
            price = fetch_google_finance(symbol)
            if price is None:
                price = 0  # fallback
            data.append({"name": name, "symbol": symbol, "price": price})
        df = pd.DataFrame(data)
        df.to_csv(INDEX_CSV, index=False)
    return df

# --- Load RSS news with CSV cache ---
def load_rss_news():
    if is_cache_fresh(RSS_CSV):
        df = pd.read_csv(RSS_CSV)
        df["published"] = pd.to_datetime(df["published"])
    else:
        df = fetch_economic_rss()
        df.to_csv(RSS_CSV, index=False)
    return df

# --- Display indices as metrics cards ---
index_df = load_index_data()
cols = st.columns(len(indices))
for i, row in index_df.iterrows():
    cols[i].markdown(
        f"""
        <div style="
            background-color:#f9f9f9;
            border-radius:10px;
            padding:15px;
            box-shadow:0 2px 4px rgba(0,0,0,0.1);
            text-align:center;">
            <h4 style="margin-bottom:5px;">{row['name']}</h4>
            <p style="margin:0;color:gray;">₹ {row['price']:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Display RSS news as cards ---
st.subheader("📰 Latest Indian Economic News")
rss_df = load_rss_news()
cols_news = st.columns(2)
for i, row in rss_df.iterrows():
    col = cols_news[i % 2]
    col.markdown(
        f"""
        <div style="
            background-color:#f9f9f9;
            border-radius:10px;
            padding:10px;
            margin-bottom:10px;
            box-shadow:0 2px 4px rgba(0,0,0,0.1);">
            <h5 style="margin-bottom:5px;">{row['title']}</h5>
            <p style="margin:0;color:gray;font-size:13px;">{row['published'].strftime('%d %b %Y %H:%M')}</p>
            <a href="{row['link']}" target="_blank">Read More</a>
        </div>
        """,
        unsafe_allow_html=True
    )
#---------------------------------------------------------------
  #----------------------PRICE TREND----------------------------
    

st.subheader("📈 Price Trend")
if not stocks_df.empty:
    fig = px.line(stocks_df[stocks_df["symbol"].isin(selected_symbols)],
                  x="timestamp", y="close", color="symbol",
                  title="Stock Closing Prices")
    st.plotly_chart(fig, use_container_width=True)

#-----------------------Stocks TOP GAINERS & LOSERS---------

# Take the latest record per symbol
latest_df = stocks_df.groupby("symbol").last().reset_index()

# Compute price change
latest_df["price_change"] = latest_df["close"] - latest_df["open"]

# 1. SIMPLIFY the function to return only the text with the arrow, NO HTML
def price_arrow_text(val):
    # This ensures the 'Change' value itself is formatted to 2 decimals
    if val > 0:
        return f"↑ {val:.2f}"
    elif val < 0:
        return f"↓ {val:.2f}"
    else:
        return f"{val:.2f}"

# 2. Function to apply color (for the Styler)
def color_change_cell(val):
    # val is the *text* from the price_arrow_text function (e.g., "↑ 1.50")
    if isinstance(val, str):
        if val.startswith('↑'):
            return 'color: green; font-weight: bold;'
        elif val.startswith('↓'):
            return 'color: red; font-weight: bold;'
    return 'color: black;'


# Sort top 5 gainers and losers
top_gainers = latest_df.sort_values("price_change", ascending=False).head(5).copy()
top_losers = latest_df.sort_values("price_change").head(5).copy()

# Apply the simplified arrow text function
top_gainers["Change"] = top_gainers["price_change"].apply(price_arrow_text)
top_losers["Change"] = top_losers["price_change"].apply(price_arrow_text)


# Display in two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Top 5 Gainers")
    
    # 3. Apply the Styler to the dataframe
    styled_gainers = (
        top_gainers[["symbol", "close", "Change"]]
        # 🌟 ADDED PRECISION FORMATTING HERE 🌟
        .style.format({'close': '{:.2f}'})
        # Apply the color to the 'Change' column
        .applymap(color_change_cell, subset=['Change'])
    )
    st.dataframe(styled_gainers, hide_index=True) # Use the styled object here

with col2:
    st.markdown("### 📉 Top 5 Losers")
    
    # 3. Apply the Styler to the dataframe
    styled_losers = (
        top_losers[["symbol", "close", "Change"]]
        # 🌟 ADDED PRECISION FORMATTING HERE 🌟
        .style.format({'close': '{:.2f}'})
        # Apply the color to the 'Change' column
        .applymap(color_change_cell, subset=['Change'])
    )
    st.dataframe(styled_losers, hide_index=True) # Use the styled object here
# ------------------------------------------------------------------------------
# BUY / SELL PREDICTIONS
# ------------------------------------------------------------------------------

# st.subheader("💹 Buy/Sell Recommendations")
# st.dataframe(pred_df[["symbol", "action","price","target_price","stop_loss","buy_pred","sell_pred"]])

import streamlit as st
import pandas as pd
# NOTE: pred_df must be loaded and available (e.g., from a CSV, database, or previous computation).

# --- 1. CSS Styling Functions ---

def color_action_cell(val):
    """
    Apply text color styling based on trade action.
    This function now colors the text which already contains the arrow.
    """
    # The arrow is now part of the string (e.g., "⬆️ Buy")
    val_str = str(val).lower()
    if "buy" in val_str:
        # Green Up Arrow (Color is applied by this return)
        return 'color: green; font-weight: bold;'
    elif "sell" in val_str:
        # Red Down Arrow (Color is applied by this return)
        return 'color: red; font-weight: bold;'
    elif "no trade" in val_str:
        # Neutral color for No Trade
        return 'color: orange; font-weight: bold;'
    return 'color: black;'

def green_text_style(val):
    """Apply green color and bold font."""
    return 'color: green; font-weight: bold;'

def red_text_style(val):
    """Apply red color and bold font."""
    return 'color: red; font-weight: bold;'

# --- 2. Data Preparation ---

# Convert to float safely (ignore if already numeric)
for col in ["buy_pred", "sell_pred", "price", "target_price", "stop_loss"]:
    if col in pred_df.columns:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

# Rename columns for clean UI
display_df = pred_df.rename(columns={
    "symbol" :"Symbol",
    "action": "Action 🚦",
    "price": "Price ₹",
    "target_price": "Target 🎯",
    "stop_loss": "Stop Loss 🛑",
    "buy_pred": "Buy % ⬆️",
    "sell_pred": "Sell % ⬇️"
}).copy() # Use .copy() to avoid SettingWithCopyWarning

# 🌟 MODIFIED FUNCTION: ADD ARROW AND NEUTRAL ICON 🌟
def add_arrow_to_action(action):
    """Prepends the correct arrow/icon for Buy/Sell/No Trade."""
    action_str = str(action).lower()
    if action_str == "buy":
        return "⬆️ Buy" # Green Up Arrow
    elif action_str == "sell":
        return "⬇️ Sell" # Red Down Arrow
    elif action_str == "no trade":
        return "⏸️ No Trade" # Neutral pause/stop icon
    return action

display_df["Action 🚦"] = display_df["Action 🚦"].apply(add_arrow_to_action)


# Select only relevant columns
display_cols = ["Symbol", "Action 🚦", "Price ₹", "Target 🎯", "Stop Loss 🛑", "Buy % ⬆️", "Sell % ⬇️"]
display_df = display_df[[col for col in display_cols if col in display_df.columns]]

# --- 3. Apply Styler ---
styled_pred_df = (
    display_df
    .style
    # 1. Color trade action (Buy/Sell/No Trade)
    .applymap(color_action_cell, subset=["Action 🚦"])
    
    # 2. Color Buy prediction column green
    .applymap(green_text_style, subset=["Buy % ⬆️"])
    
    # 3. Color Sell prediction column red
    .applymap(red_text_style, subset=["Sell % ⬇️"])
    
    # 4. Apply formatting (including Rupee symbol)
    .format({
        "Price ₹": "₹ {:.2f}",
        "Target 🎯": "₹ {:.2f}",
        "Stop Loss 🛑": "₹ {:.2f}",
        "Buy % ⬆️": "{:.2f}",
        "Sell % ⬇️": "{:.2f}"
    })
)

# --- 4. Display in Streamlit ---
st.subheader("💹 ML Trade Recommendations")
st.dataframe(styled_pred_df, hide_index=True, use_container_width=True)
#-----------------------------------------------------
##------------------------------------------------------------------------
##NEWS HEADLINES______
############################################################################
#----------
st.subheader("📰 Latest News & Sentiment")

# Assuming DB_URI and selected_symbols are defined elsewhere in the Streamlit app's scope
engine = create_engine(DB_URI)  # Make sure DB_URI is correct

def fetch_news(symbols=None, top_n=5):
    """Fetch news with internal stock_date for ordering (not displayed)"""
    if symbols is None or len(symbols) == 0:
        # general news
        query = f"""
        SELECT n.symbol, n.title, n.sentiment, MAX(s.timestamp) AS stock_date
        FROM news_sentiment n
        JOIN stocks s ON n.symbol = s.symbol
        GROUP BY n.symbol, n.title, n.sentiment
        ORDER BY stock_date DESC
        LIMIT 3;
        """
    else:
        symbols_list = ",".join([f"'{s.upper()}'" for s in symbols])
        query = f"""
        WITH ranked_news AS (
            SELECT
                s.symbol,
                n.title,
                n.sentiment,
                s.timestamp AS stock_date,
                ROW_NUMBER() OVER (PARTITION BY s.symbol ORDER BY s.timestamp DESC) AS rn
            FROM stocks s
            JOIN news_sentiment n
              ON s.symbol = n.symbol
            WHERE s.symbol IN ({symbols_list})
        )
        SELECT symbol, title, sentiment, stock_date
        FROM ranked_news
        WHERE rn <= {top_n}
        ORDER BY stock_date DESC;
        """
    return pd.read_sql(query, engine)

def map_sentiment(sent):
    sent = str(sent).lower()
    if sent in ["positive", "bullish"]:
        return "Bullish"
    elif sent in ["negative", "bearish"]:
        return "Bearish"
    else:
        return "Neutral"

def color_sentiment(val):
    """Apply CSS text color based on sentiment value."""
    if val == "Bullish":
        return "color: green; font-weight: bold;"  # Green text for Bullish
    elif val == "Bearish":
        return "color: red; font-weight: bold;"    # Red text for Bearish
    else:
        return "color: black;"                     # Black text for Neutral

# Fetch stock-specific news
# Using 'locals()' is a heuristic way to check if selected_symbols is defined in the script's global scope
if 'selected_symbols' in locals() and selected_symbols:
    if len(selected_symbols) == 1:
        stock_news_df = fetch_news(selected_symbols, top_n=5)
    else:
        stock_news_df = fetch_news(selected_symbols, top_n=1)
else:
    stock_news_df = pd.DataFrame(columns=["symbol","title","sentiment","stock_date"])

# Fetch general news (2–3)
general_news_df = fetch_news(top_n=3)

# Combine all news
news_df = pd.concat([stock_news_df, general_news_df], ignore_index=True)
if news_df.empty:
    st.info("No news available.")
else:
    # Map sentiment
    news_df["Sentiment"] = news_df["sentiment"].apply(map_sentiment)
    
    # 🌟 RENAME COLUMNS FOR FINAL DISPLAY 🌟
    news_df_display = news_df.rename(columns={
        "symbol": "Symbol",
        "title": "News Headline",
        "Sentiment": "Sentiment 🧠"
    })
    
    # Select final display columns
    news_df_display = news_df_display[["Symbol", "News Headline", "Sentiment 🧠"]]

    # Display color-coded table
    st.dataframe(
        news_df_display.style.map(color_sentiment, subset=["Sentiment 🧠"]), 
        use_container_width=True,
        hide_index=True
    )
#------------------------------------------
# # ------------------------------------------------------------------------------


