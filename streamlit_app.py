import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz


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
engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

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

# --------------------------------------------------------------------
# 🕒 MARKET STATUS INDICATOR
# --------------------------------------------------------------------

# Define Indian timezone
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist)
today = now_ist.date()

# Define market open/close times (IST)
market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

# Check if market is open
is_weekday = now_ist.weekday() < 5  # Monday=0, Friday=4
is_open_time = market_open <= now_ist <= market_close
market_open_now = is_weekday and is_open_time

# Compute last Friday
days_since_friday = (now_ist.weekday() - 4) % 7
last_friday = now_ist - timedelta(days=days_since_friday)
last_friday_date = last_friday.strftime("%Y-%m-%d")

# Display top-right indicator using Streamlit columns
col1, col2 = st.columns([4, 1])

with col2:
    st.markdown(
        f"""
        <div style="text-align: right; font-size: 16px;">
            <b>{now_ist.strftime("%A, %d %B %Y")}</b><br>
            <span style="color:{'green' if market_open_now else 'red'}; font-weight:bold;">
                {'🟢 Market Open' if market_open_now else '🔴 Market Closed'}
            </span><br>
            <span style="font-size:13px;">{now_ist.strftime('%I:%M %p')} IST</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------------------------
# 🧭 DATA FILTERING LOGIC (show Friday data if market closed)
# --------------------------------------------------------------------
if not market_open_now:
    st.warning(f"🛑 Market is closed. Showing last Friday's data ({last_friday_date}).")

    # Filter stock data to show only up to last Friday
    try:
        stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])
        stocks_df = stocks_df[stocks_df["timestamp"].dt.date == pd.Timestamp(last_friday_date).date()]
    except Exception as e:
        st.error(f"Error filtering Friday data: {e}")


        
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

    import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="Indian Stock Events", page_icon="📅", layout="wide")

# --------------------------
# DATA: Stock Holidays
# --------------------------
holidays = pd.DataFrame([
    {"date": "2025-10-02", "day": "Thursday", "holiday": "Mahatma Gandhi Jayanti / Dussehra"},
    {"date": "2025-10-21", "day": "Tuesday", "holiday": "Diwali Laxmi Pujan (Muhurat Trading)"},
    {"date": "2025-10-22", "day": "Wednesday", "holiday": "Diwali-Balipratipada"},
    {"date":"2025-11-05", "day":"Wednesday", "holiday":"Prakash Gurpurb Sri Guru Nanak Dev"},
    {"date": "2025-12-25", "day": "Thursday", "holiday": "Christmas Day"},
])
holidays["date"] = pd.to_datetime(holidays["date"])

# --------------------------
# DATA: Quarterly Financial Results
# --------------------------
financial_events = pd.DataFrame([
    {"date": "2025-10-15", "company": "Chemicals Corp.", "event": "Board Meeting: New Project Sanction"},
    {"date": "2025-10-20", "company": "Logistics Pro", "event": "Dividend: Interim Dividend Declaration"},
    {"date": "2025-10-25", "company": "EduTech Global", "event": "Annual General Meeting"},
    {"date": "2025-10-30", "company": "Real Estate Developers", "event": "EGM: Fundraising Approval"},
    {"date": "2025-11-02", "company": "Tech Solutions Ltd.", "event": "Financial Results: Q3 Earnings"},
    {"date": "2025-11-05", "company": "Consumer Goods Ltd.", "event": "Board Meeting: Marketing Budget Review"},
    {"date": "2025-11-10", "company": "Global Pharma Inc.", "event": "Dividend: Final Dividend Record Date"},
    {"date": "2025-11-25", "company": "Power Grid Solutions", "event": "Board Meeting: Expansion Project Review"},
    {"date": "2025-12-01", "company": "Textile Mills India", "event": "Financial Results: Half-Yearly Financials"},
    {"date": "2025-12-05", "company": "Telecom Connect", "event": "Annual General Meeting"},
    {"date": "2025-12-10", "company": "Food & Beverages", "event": "Dividend: Special Dividend Declaration"},
])
financial_events["date"] = pd.to_datetime(financial_events["date"])

# --------------------------
# FILTER: Only future events
# --------------------------
today = datetime.now().date()
financial_events_upcoming = financial_events[financial_events["date"].dt.date >= today].sort_values("date")
holidays_upcoming = holidays[holidays["date"].dt.date >= today].sort_values("date")

# --------------------------
# FUNCTION: Days Countdown
# --------------------------
def days_until(event_date):
    return (event_date.date() - today).days

financial_events_upcoming["Days Left"] = financial_events_upcoming["date"].apply(days_until)
holidays_upcoming["Days Left"] = holidays_upcoming["date"].apply(days_until)

# --------------------------
# SIDEBAR: Show Events in Boxes
# --------------------------
st.sidebar.header("📅 Upcoming Events")

# Financial Events Box
st.sidebar.markdown("### 💹 Quarterly Financial Events")
for _, row in financial_events_upcoming.iterrows():
    st.sidebar.markdown(
        f"""
        <div style='border:2px solid #4CAF50; padding:10px; border-radius:10px; margin-bottom:10px; background-color:#e8f5e9'>
        <b>{row['event']}</b> for {row['company']}<br>
        📅 {row['date'].date()} ({row['date'].strftime('%A')})<br>
        ⏳ {row['Days Left']} days left
        </div>
        """, unsafe_allow_html=True
    )

# Holidays Box
st.sidebar.markdown("### 🎉 Stock Market Holidays")
for _, row in holidays_upcoming.iterrows():
    st.sidebar.markdown(
        f"""
        <div style='border:2px solid #2196F3; padding:10px; border-radius:10px; margin-bottom:10px; background-color:#e3f2fd'>
        <b>{row['holiday']}</b><br>
        📅 {row['date'].date()} ({row['day']})<br>
        ⏳ {row['Days Left']} days left
        </div>
        """, unsafe_allow_html=True
    )


# --------------------------
# MAIN + CHAT LAYOUT
# --------------------------
col_main, col_chat = st.columns([3, 1])  # 75% main, 25% chat
# --------------------------
# MAIN DASHBOARD
# --------------------------
with col_main:
    st.title("📊 Indian Stock Dashboard")

    # ----------------------------------------------
# BUY/SELL PREDICTION SECTION WITH DROPDOWN
# ----------------------------------------------

st.subheader("💬Stock Buy/Sell Recommendation: ")

# Get all unique symbols from stocks DataFrame
all_symbols = stocks_df["symbol"].unique().tolist()

# Create a dropdown to select stock
selected_stock = st.selectbox("Select a stock:", options=all_symbols)

# Once a stock is selected, fetch its prediction
if selected_stock:
    rec = pred_df[pred_df["symbol"] == selected_stock]

    if not rec.empty:
        buy_pred = rec.iloc[0]["buy_pred"]
        sell_pred = rec.iloc[0]["sell_pred"]
        action = rec.iloc[0]["action"]

        st.success(
            f"**{selected_stock}** → Model suggests **{action}** "
            f"(Buy confidence: {buy_pred*100:.1f}%, Sell confidence: {sell_pred*100:.1f}%)"
        )
    else:
        st.info(f"No prediction available for {selected_stock}.")

#--------------------Price Trend--------------------------------    
# st.subheader("📈 Price Trend")
if not stocks_df.empty:
    fig = px.line(stocks_df[stocks_df["symbol"].isin(selected_symbols)],
                  x="timestamp", y="close", color="symbol",
                  title="Stock Closing Prices")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Compute percentage change
# -------------------------
stocks_df["price_change_pct"] = ((stocks_df["close"] - stocks_df["open"]) / stocks_df["open"]) * 100

# -------------------------
# Functions for arrow and color
# -------------------------
def price_arrow_text_pct(val):
    if val > 0:
        return f"↑ {val:.2f}%"
    elif val < 0:
        return f"↓ {abs(val):.2f}%"
    else:
        return f"{val:.2f}%"

def color_change_cell(val):
    if isinstance(val, str):
        if val.startswith("↑"):
            return "color: green; font-weight: bold;"
        elif val.startswith("↓"):
            return "color: red; font-weight: bold;"
    return "color: black;"

# -------------------------
# Sort top 5 gainers and losers
# -------------------------
top_gainers = stocks_df.sort_values("price_change_pct", ascending=False).head(5).copy()
top_losers = stocks_df.sort_values("price_change_pct").head(5).copy()

# Apply arrow text
top_gainers["Change %"] = top_gainers["price_change_pct"].apply(price_arrow_text_pct)
top_losers["Change %"] = top_losers["price_change_pct"].apply(price_arrow_text_pct)

# Rename columns for display
display_gainers = top_gainers[["symbol", "Change %"]].rename(
    columns={"symbol": "Stock Name", "Change %": "Price Change (%)"}
)
display_losers = top_losers[["symbol", "Change %"]].rename(
    columns={"symbol": "Stock Name", "Change %": "Price Change (%)"}
)

# -------------------------
# Display in Streamlit
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Top 5 Gainers")
    styled_gainers = display_gainers.style.applymap(color_change_cell, subset=["Price Change (%)"])
    st.dataframe(styled_gainers, hide_index=True, use_container_width=True)

with col2:
    st.markdown("### 📉 Top 5 Losers")
    styled_losers = display_losers.style.applymap(color_change_cell, subset=["Price Change (%)"])
    st.dataframe(styled_losers, hide_index=True, use_container_width=True)

#-----------------------Stocks TOP GAINERS & LOSERS---------

# # Take the latest record per symbol
# latest_df = stocks_df.groupby("symbol").last().reset_index()

# # Compute percentage change
# latest_df["price_change_pct"] = ((latest_df["close"] - latest_df["open"]) / latest_df["open"]) * 100

# # Function to return arrow with percentage
# def price_arrow_text_pct(val):
#     if val > 0:
#         return f"↑ {val:.2f}%"
#     elif val < 0:
#         return f"↓ {abs(val):.2f}%"
#     else:
#         return f"{val:.2f}%"

# # Function to apply color
# def color_change_cell(val):
#     if isinstance(val, str):
#         if val.startswith('↑'):
#             return 'color: green; font-weight: bold;'
#         elif val.startswith('↓'):
#             return 'color: red; font-weight: bold;'
#     return 'color: black;'

# # Sort top 5 gainers and losers by percentage
# top_gainers = latest_df.sort_values("price_change_pct", ascending=False).head(5).copy()
# top_losers = latest_df.sort_values("price_change_pct").head(5).copy()

# # Apply arrow text
# top_gainers["Change %"] = top_gainers["price_change_pct"].apply(price_arrow_text_pct)
# top_losers["Change %"] = top_losers["price_change_pct"].apply(price_arrow_text_pct)

# # Display in two columns
# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("### 📈 Top 5 Gainers")
#     styled_gainers = (
#         top_gainers[["symbol", "close", "Change %"]]
#         .style.format({'close': '{:.2f}'})
#         .applymap(color_change_cell, subset=['Change %'])
#     )
#     st.dataframe(styled_gainers, hide_index=True, use_container_width=True)

# with col2:
#     st.markdown("### 📉 Top 5 Losers")
#     styled_losers = (
#         top_losers[["symbol", "close", "Change %"]]
#         .style.format({'close': '{:.2f}'})
#         .applymap(color_change_cell, subset=['Change %'])
#     )
#     st.dataframe(styled_losers, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# BUY / SELL PREDICTIONS
# ------------------------------------------------------------------------------
import streamlit as st
import pandas as pd

# -------------------------
# NOTE: pred_df must be preloaded (from CSV, database, or computation)
# -------------------------

# --- 1. CSS Styling Functions ---
def color_action_cell(val):
    """Color Buy/Sell/No Trade actions with arrows/icons."""
    val_str = str(val).lower()
    if "buy" in val_str:
        return 'color: green; font-weight: bold;'
    elif "sell" in val_str:
        return 'color: red; font-weight: bold;'
    elif "no trade" in val_str:
        return 'color: orange; font-weight: bold;'
    return 'color: black;'

def green_text_style(val):
    return 'color: green; font-weight: bold;'

def red_text_style(val):
    return 'color: red; font-weight: bold;'

# --- 2. Data Preparation ---
# Convert numeric columns safely
for col in ["buy_pred", "sell_pred", "price", "target_price", "stop_loss"]:
    if col in pred_df.columns:
        pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

# Rename columns for UI
display_df = pred_df.rename(columns={
    "symbol": "Symbol",
    "action": "Action 🚦",
    "price": "Price ₹",
    "target_price": "Target 🎯",
    "stop_loss": "Stop Loss 🛑",
    "buy_pred": "Buy % ⬆️",
    "sell_pred": "Sell % ⬇️"
}).copy()

# Add arrow/icons to Action
def add_arrow_to_action(action):
    action_str = str(action).lower()
    if action_str == "buy":
        return "⬆️ Buy"
    elif action_str == "sell":
        return "⬇️ Sell"
    elif action_str == "no trade":
        return "⏸️ No Trade"
    return action

display_df["Action 🚦"] = display_df["Action 🚦"].apply(add_arrow_to_action)

# Convert Buy/Sell predictions to percentages
display_df["Buy % ⬆️"] = display_df["Buy % ⬆️"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
display_df["Sell % ⬇️"] = display_df["Sell % ⬇️"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")

# Select relevant columns
display_cols = ["Symbol", "Action 🚦", "Price ₹", "Target 🎯", "Stop Loss 🛑", "Buy % ⬆️", "Sell % ⬇️"]
display_df = display_df[[col for col in display_cols if col in display_df.columns]]

# --- 3. Apply Styler ---
styled_pred_df = (
    display_df.style
    .applymap(color_action_cell, subset=["Action 🚦"])
    .applymap(green_text_style, subset=["Buy % ⬆️"])
    .applymap(red_text_style, subset=["Sell % ⬇️"])
    .format({
        "Price ₹": "₹ {:.2f}",
        "Target 🎯": "₹ {:.2f}",
        "Stop Loss 🛑": "₹ {:.2f}"
    })
)

# --- 4. Display in Streamlit ---
st.subheader("💹 ML Trade Recommendations")
st.dataframe(styled_pred_df, hide_index=True, use_container_width=True)

# # st.subheader("💹 Buy/Sell Recommendations")
# # st.dataframe(pred_df[["symbol", "action","price","target_price","stop_loss","buy_pred","sell_pred"]])

# import streamlit as st
# import pandas as pd
# # NOTE: pred_df must be loaded and available (e.g., from a CSV, database, or previous computation).

# # --- 1. CSS Styling Functions ---

# def color_action_cell(val):
#     """
#     Apply text color styling based on trade action.
#     This function now colors the text which already contains the arrow.
#     """
#     # The arrow is now part of the string (e.g., "⬆️ Buy")
#     val_str = str(val).lower()
#     if "buy" in val_str:
#         # Green Up Arrow (Color is applied by this return)
#         return 'color: green; font-weight: bold;'
#     elif "sell" in val_str:
#         # Red Down Arrow (Color is applied by this return)
#         return 'color: red; font-weight: bold;'
#     elif "no trade" in val_str:
#         # Neutral color for No Trade
#         return 'color: orange; font-weight: bold;'
#     return 'color: black;'

# def green_text_style(val):
#     """Apply green color and bold font."""
#     return 'color: green; font-weight: bold;'

# def red_text_style(val):
#     """Apply red color and bold font."""
#     return 'color: red; font-weight: bold;'

# # --- 2. Data Preparation ---

# # Convert to float safely (ignore if already numeric)
# for col in ["buy_pred", "sell_pred", "price", "target_price", "stop_loss"]:
#     if col in pred_df.columns:
#         pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")

# # Rename columns for clean UI
# display_df = pred_df.rename(columns={
#     "symbol" :"Symbol",
#     "action": "Action 🚦",
#     "price": "Price ₹",
#     "target_price": "Target 🎯",
#     "stop_loss": "Stop Loss 🛑",
#     "buy_pred": "Buy % ⬆️",
#     "sell_pred": "Sell % ⬇️"
# }).copy() # Use .copy() to avoid SettingWithCopyWarning

# # 🌟 MODIFIED FUNCTION: ADD ARROW AND NEUTRAL ICON 🌟
# def add_arrow_to_action(action):
#     """Prepends the correct arrow/icon for Buy/Sell/No Trade."""
#     action_str = str(action).lower()
#     if action_str == "buy":
#         return "⬆️ Buy" # Green Up Arrow
#     elif action_str == "sell":
#         return "⬇️ Sell" # Red Down Arrow
#     elif action_str == "no trade":
#         return "⏸️ No Trade" # Neutral pause/stop icon
#     return action

# display_df["Action 🚦"] = display_df["Action 🚦"].apply(add_arrow_to_action)


# # Select only relevant columns
# display_cols = ["Symbol", "Action 🚦", "Price ₹", "Target 🎯", "Stop Loss 🛑", "Buy % ⬆️", "Sell % ⬇️"]
# display_df = display_df[[col for col in display_cols if col in display_df.columns]]

# # --- 3. Apply Styler ---
# styled_pred_df = (
#     display_df
#     .style
#     # 1. Color trade action (Buy/Sell/No Trade)
#     .applymap(color_action_cell, subset=["Action 🚦"])
    
#     # 2. Color Buy prediction column green
#     .applymap(green_text_style, subset=["Buy % ⬆️"])
    
#     # 3. Color Sell prediction column red
#     .applymap(red_text_style, subset=["Sell % ⬇️"])
    
#     # 4. Apply formatting (including Rupee symbol)
#     .format({
#         "Price ₹": "₹ {:.2f}",
#         "Target 🎯": "₹ {:.2f}",
#         "Stop Loss 🛑": "₹ {:.2f}",
#         "Buy Confidence % ⬆️": "{:.2f}",
#         "Sell % ⬇️": "{:.2f}"
#     })
# )

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
# # # ------------------------------------------------------------------------------
# # # BUY/SELL PREDICTION
# # # ------------------------------------------------------------------------------

# st.subheader("💬 BUY/SELL PREDICTION")
# query = st.text_input("Ask about any stock:")
# if query:
#     query = query.upper()
#     if query in all_symbols:
#         rec = pred_df[pred_df["symbol"] == query]
#         if not rec.empty:
#             buy_pred = rec.iloc[0]["buy_pred"]
#             sell_pred = rec.iloc[0]["sell_pred"]
#             action = rec.iloc[0]["action"]
#             st.success(f"{query}: Model suggests **{action}** (with buy confidence {buy_pred*100:.1f}% and sell confidence {sell_pred*100:.1f}%)")
#         else:
#             st.info(f"No prediction available for {query}.")
#     else:
#         st.info("Please type a valid stock symbol (e.g. TCS, INFY).")

# --------------------------------------------------------------------
# 📒 TRADEBOOK DATA
# --------------------------------------------------------------------
# --------------------------
#LOAD TRADEBOOK
#--------------------------
st.subheader("💼 Tradebook Summary")

try:
    engine = create_engine(DB_URI)
    tradebook_df = pd.read_sql("SELECT * FROM tradebook", con=engine)

    if not tradebook_df.empty:
        # Rename key columns if they exist
        rename_dict = {}
        if "price" in tradebook_df.columns:
            rename_dict["price"] = "entry_price"
        if "ltp" in tradebook_df.columns:
            rename_dict["ltp"] = "exit_price"
        if "pnl_percent" in tradebook_df.columns:
            rename_dict["pnl_percent"] = "profit_loss_percent"

        tradebook_df = tradebook_df.rename(columns=rename_dict)

        # Ensure numeric formatting
        for col in ["entry_price", "exit_price", "profit_loss_percent"]:
            if col in tradebook_df.columns:
                tradebook_df[col] = pd.to_numeric(tradebook_df[col], errors="coerce")

        # Style function for P&L coloring
        def color_profit(val):
            if pd.isna(val):
                return ""
            if val > 0:
                return "color: green; font-weight: bold;"
            elif val < 0:
                return "color: red; font-weight: bold;"
            return "color: black;"

        # Columns to display (including symbol if present)
        display_cols = []
        if "symbol" in tradebook_df.columns:
            display_cols.append("symbol")
        for col in ["entry_price", "exit_price", "profit_loss_percent"]:
            if col in tradebook_df.columns:
                display_cols.append(col)

        # Display table
        styled_tradebook = tradebook_df[display_cols].style.format({
            "entry_price": "₹ {:.2f}",
            "exit_price": "₹ {:.2f}",
            "profit_loss_percent": "{:.2f}%"
        }).applymap(color_profit, subset=[col for col in ["profit_loss_percent"] if col in display_cols])

        st.dataframe(styled_tradebook, use_container_width=True, hide_index=True)
        st.success(f"✅ Loaded {len(tradebook_df)} trade records.")
    else:
        st.info("No trades found in the tradebook table.")

except Exception as e:
    st.error(f"⚠️ Could not load tradebook data: {e}")

# --------------------------
# Predicted Tradebook
# --------------------------
# st.subheader("💼 Predicted Tradebook")

# try:
#     engine = create_engine(DB_URI)
#     tradebook_df = pd.read_sql('SELECT * FROM "tradebook"', con=engine)

#     if not tradebook_df.empty:

#         # Ensure numeric columns
#         if "pnl_percent" in tradebook_df.columns:
#             tradebook_df["pnl_percent"] = pd.to_numeric(tradebook_df["pnl_percent"], errors="coerce")
#         else:
#             tradebook_df["pnl_percent"] = 0

#         # Short explanation column
#         def trade_explanation(row):
#             action = str(row.get('action', '')).lower()
#             pnl = row.get('pnl_percent', 0)

#             if action == 'buy' and pnl > 0:
#                 return "Bought → Profit"
#             elif action == 'buy' and pnl < 0:
#                 return "Bought → Loss"
#             elif action == 'sell' and pnl > 0:
#                 return "Sold → Profit"
#             elif action == 'sell' and pnl < 0:
#                 return "Sold → Loss"
#             elif action in ['hold', 'no trade', '']:
#                 return "No trade"
#             else:
#                 return "Trade Closed"

#         tradebook_df["Explanation"] = tradebook_df.apply(trade_explanation, axis=1)

#         # Columns to display (including symbol if present)
#         display_cols = []
#         if "symbol" in tradebook_df.columns:
#             display_cols.append("symbol")
#         if "action" in tradebook_df.columns:
#             display_cols.append("action")
#         display_cols.append("pnl_percent")
#         display_cols.append("Explanation")

#         # Style function for P&L coloring
#         def color_profit(val):
#             if pd.isna(val):
#                 return ""
#             if val > 0:
#                 return "color: green; font-weight: bold;"
#             elif val < 0:
#                 return "color: red; font-weight: bold;"
#             return "color: black;"

#         styled_tradebook = tradebook_df[display_cols].style.format({
#             "pnl_percent": "{:.2f}%"
#         }).applymap(color_profit, subset=["pnl_percent"])

#         st.dataframe(styled_tradebook, use_container_width=True, hide_index=True)
#         st.success(f"✅ Loaded {len(tradebook_df)} predicted trades.")

#     else:
#         st.info("No predicted trades found in the tradebook.")

# except Exception as e:
#     st.error(f"⚠️ Could not load predicted tradebook: {e}")