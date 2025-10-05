import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import requests


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

# --------------------------
# MARKET DATE LOGIC
# --------------------------
today = datetime.now().date()
if today.weekday() >= 5:  # Sat/Sun
    last_friday = today - timedelta(days=today.weekday() - 4)
    st.warning(f"Market closed 🛑 Showing last Friday ({last_friday}) data")
    stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])
    stocks_df = stocks_df[stocks_df["timestamp"] <= pd.Timestamp(last_friday)]

# --------------------------
# SIDEBAR CONTROLS
# --------------------------
st.sidebar.header("⚙️ Controls")
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

    # # CHATBOT (basic rule-based)
# # ------------------------------------------------------------------------------

st.subheader("Stock Buy/Sell Recommender")
query = st.text_input("Ask about any stock eg TCS,BELL etc:")
if query:
    query = query.upper()
    if query in all_symbols:
        rec = pred_df[pred_df["symbol"] == query]
        if not rec.empty:
            buy_pred = rec.iloc[0]["buy_pred"]
            sell_pred = rec.iloc[0]["sell_pred"]
            action = rec.iloc[0]["action"]
            st.success(f"{query}: Model suggests **{action}** (with buy confidence {buy_pred*100:.1f}% and sell confidence {sell_pred*100:.1f}%)")
        else:
            st.info(f"No prediction available for {query}.")
    else:
        st.info("Please type a valid stock symbol (e.g. TCS, INFY).")

    
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
st.subheader("💹 Buy/Sell Recommendations")
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


