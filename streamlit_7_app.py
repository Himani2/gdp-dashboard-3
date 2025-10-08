import streamlit as st
import pandas as pd
import os
import pytz
import numpy as np
from datetime import datetime, time, timedelta
from sqlalchemy import create_engine
# altair import removed for lightweight dashboard

# =====================================
# CONFIGURATION
# =====================================
# NOTE: Placeholder variables for demonstration. 
# You MUST replace these with your actual PostgreSQL credentials 
# to enable live data fetching.
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"

DB_PORT = "5432" 
DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{DB_PORT}/{db_name}'

# Market Timings
MARKET_OPEN_TIME = time(9, 15)
MARKET_CLOSE_TIME = time(15, 30)
IST = pytz.timezone("Asia/Kolkata")

# =====================================
# DATABASE & CACHE SETUP
# =====================================
DB_CONNECTED = False
try:
    if db_user and db_password and db_host and db_name:
        engine = create_engine(DB_URI, connect_args={'connect_timeout':5})
        with engine.connect():
            DB_CONNECTED = True
    else:
        pass
except Exception:
    DB_CONNECTED = False

DATA_PATH = "./cache"
os.makedirs(DATA_PATH, exist_ok=True)
CACHE_TTL = 300 # Cache duration in seconds

@st.cache_data(ttl=CACHE_TTL)
def load_table(table_name):
    """Loads data from DB or cache file."""
    cache_file = f"{DATA_PATH}/{table_name}.csv"
    df = None
    if DB_CONNECTED:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        except Exception:
            pass # Use cache if DB query fails

    if df is not None and not df.empty:
        # Save to cache and return
        df.to_csv(cache_file, index=False)
        return df
    
    if os.path.exists(cache_file):
        # Load from cache if DB is not connected or failed
        try:
            return pd.read_csv(cache_file)
        except:
            return pd.DataFrame()
            
    # Fallback to empty DataFrame if no data found
    return pd.DataFrame()

def ensure_data_fallback(df, ist_tz):
    """
    Ensures a basic, single-snapshot DataFrame exists for calculations 
    if the database load failed completely, avoiding complex history simulation. 
    This keeps the dashboard lightweight by only providing minimal dummy data.
    """
    if df.empty or 'symbol' not in df.columns:
         # Create dummy data if everything is empty
         symbols = ['TCS', 'RELIANCE', 'HDFC', 'INFY', 'ICICI']
         now = datetime.now(ist_tz)
         return pd.DataFrame({
             'symbol': symbols,
             'close': np.random.uniform(100, 1000, size=len(symbols)),
             'open': np.random.uniform(100, 1000, size=len(symbols)),
             'timestamp': [now] * len(symbols)
         })
    return df

# Load DataFrames
stocks_df = load_table("stocks")
buy_sell_df = load_table("buy_sell_predictions")
news_df = load_table("news_sentiment")

# Ensure stocks_df has minimal data if DB/cache load failed
# This replaces the heavier simulation logic previously used for the removed chart.
stocks_df = ensure_data_fallback(stocks_df, IST)


# =====================================
# MARKET STATUS & HOLIDAY LOGIC
# =====================================

def get_indian_holidays(year):
    """Returns a dictionary of sample Indian stock market holidays (NSE/BSE) for the given year."""
    # Updated sample holidays for a more realistic future calendar
    holidays = {
        datetime(year, 11, 1).date(): "Diwali (Laxmi Pujan)",
        datetime(year, 12, 25).date(): "Christmas Day",
    }
    # Add a few holidays for the subsequent year 
    holidays.update({
        datetime(year + 1, 1, 26).date(): "Republic Day",
        datetime(year + 1, 3, 29).date(): "Holi",
        datetime(year + 1, 4, 17).date(): "Ram Navami",
        datetime(year + 1, 5, 1).date(): "Maharashtra Day",
    })
    return holidays

def is_indian_stock_holiday(date):
    """Checks if the given date is a stock market holiday."""
    holidays = get_indian_holidays(date.year)
    return date in holidays

def get_market_status(now_ist):
    """Determines if the market is Open, Closed, or Holiday."""
    is_weekday = now_ist.weekday() < 5  # Monday=0 to Friday=4
    current_time = now_ist.time()
    is_open_time = (current_time >= MARKET_OPEN_TIME) and \
                   (current_time <= MARKET_CLOSE_TIME)
    is_holiday = is_indian_stock_holiday(now_ist.date())

    if is_holiday or not is_weekday:
        return "Holiday", "🟡 Market Holiday", "rgba(255, 165, 0, 0.1)", "#FFA500"
    
    if is_weekday and is_open_time:
        return "Open", "🟢 Market Open", "rgba(0, 255, 0, 0.1)", "#00FF00"
    
    return "Closed", "🔴 Market Closed", "rgba(255, 0, 0, 0.1)", "#FF0000"

def get_upcoming_holidays(now_ist):
    """Finds the next 3 upcoming holidays and calculates days away."""
    today = now_ist.date()
    current_year = today.year
    
    # Check holidays for current and next year
    all_holidays = get_indian_holidays(current_year)
    all_holidays.update(get_indian_holidays(current_year + 1))
    
    upcoming_holidays = []
    
    # Sort holidays and filter for future dates
    for date, name in sorted(all_holidays.items()):
        if date > today:
            days_away = (date - today).days
            upcoming_holidays.append((date, name, days_away))
            if len(upcoming_holidays) >= 3:
                break
    return upcoming_holidays


# =====================================
# DATA PROCESSING FOR DISPLAY
# =====================================

def preprocess_stocks(df, selected_symbols=None):
    """Calculates change percentage and filters by selected symbols.
       Processes data down to the latest snapshot per symbol.
       
       Note: We keep the .copy() inside here to ensure the function is pure
       and doesn't side-effect the cached global DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Get the latest data point for each symbol (required for current metrics)
        df = df.sort_values('timestamp').groupby('symbol').last().reset_index()
    
    # Filter for selected symbols if provided
    if selected_symbols:
        df = df[df["symbol"].isin(selected_symbols)]
    
    if 'close' in df.columns and 'open' in df.columns:
        # Calculate price change based on open/close of the latest interval
        df["price_change"] = df["close"] - df["open"]
        df["change_pct"] = (df["price_change"] / df["open"]) * 100
        
    return df

# =====================================
# STREAMLIT UI SETUP
# =====================================

# Inject Custom CSS for modern look
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #1E90FF; /* Accent color */
    }
    .stMetric label {
        color: #888;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    .prediction-header {
        font-weight: bold;
        padding-bottom: 5px;
        border-bottom: 2px solid #1E90FF; /* Stronger accent line */
        margin-bottom: 5px;
        color: #1E90FF;
    }
    </style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR CONTROLS
# =====================================
st.sidebar.header("⚙️ Dashboard Controls")

# Determine current IST and date
now_ist = datetime.now(IST)
today_date = now_ist.date()

# Stock Selector - Used for the News Filter
symbols = sorted(stocks_df["symbol"].unique()) if not stocks_df.empty and "symbol" in stocks_df.columns else []
selected_symbols = st.sidebar.multiselect(
    "Select Stocks for News Filter", symbols, default=symbols[:3] if symbols else []
)

# Holiday Notice for TODAY
st.sidebar.markdown("---")
if is_indian_stock_holiday(today_date):
    st.sidebar.info(f"**Holiday Notice:** Today ({today_date.strftime('%B %d')}) is a Market Holiday! 🏖️")
else:
    st.sidebar.success("Indian Stock Market is trading today.")

# Upcoming Holidays 
st.sidebar.subheader("🗓️ Upcoming Holidays")
upcoming_hols = get_upcoming_holidays(now_ist)

if upcoming_hols:
    for date, name, days_away in upcoming_hols:
        holiday_text = f"**{name}** in **{days_away}** days. <span style='font-size:10px;'>({date.strftime('%b %d')})</span>"
        if days_away == 1:
            holiday_text = f"**{name}** (Tomorrow!) <span style='font-size:10pt;'>({date.strftime('%b %d')})</span>"
        
        st.sidebar.markdown(holiday_text, unsafe_allow_html=True)
else:
    st.sidebar.info("No upcoming holidays found in the next year.")

# Refresh Button - Clears cache and reloads, updating top movers
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data", help="Clears cache and reloads data from the database."):
    st.cache_data.clear()
    st.rerun() 


# =====================================
# HEADER: Title, Status, and Notification Bell (Replaced popover with expander)
# =====================================
status, status_text, _, status_color = get_market_status(now_ist)

# Removed col_bell, distributing space between title and status/time
col_title, col_status_time = st.columns([6, 4])

with col_title:
    st.markdown("<h1 style='color:#1E90FF;'>FinSight Dashboard 📈</h1>", unsafe_allow_html=True)
    
with col_status_time:
    # Custom HTML for Market Status and Time Stamp
    status_bg = "white" 
    st.markdown(f"""
        <div style="text-align:right; padding:8px 15px; border-radius:10px; background-color:{status_bg}; border: 1px solid #ddd; height:100%; display:flex; flex-direction:column; justify-content:center;">
            <p style="margin:0; font-size:16px; font-weight:bold; color:{status_color};">{status_text}</p>
            <p style="margin:0; font-size:12px; color:gray;">{now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}</p>
        </div>
    """, unsafe_allow_html=True)


# Use st.expander for a stable, lightweight notification mechanism
# This is placed full-width right below the header row
with st.expander("🔔 Real-time Market Snapshot: Top Movers & News", expanded=False):

    if not stocks_df.empty:
        # Calculate next update time based on cache TTL (300 seconds)
        next_update_time = now_ist + timedelta(seconds=CACHE_TTL)
        st.markdown(f"<p style='margin:0; font-size:12px; color:gray;'>Last Updated: {now_ist.strftime('%H:%M:%S IST')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='margin:0; font-size:12px; color:gray;'>Next Cache Update: {next_update_time.strftime('%H:%M:%S IST')}</p>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

        # Process all stocks for market insights (latest data only)
        # Using the same data flow but inside the expander
        df_processed_latest = preprocess_stocks(stocks_df, stocks_df["symbol"].unique().tolist()) 
        
        # 1. Get Top 2 Market Gainers/Losers
        market_gainers = df_processed_latest.nlargest(2, "change_pct")[["symbol", "change_pct"]]
        market_losers = df_processed_latest.nsmallest(2, "change_pct")[["symbol", "change_pct"]]
        
        # 2. Get Top 2 News of the hour (Simulated: latest news)
        latest_news_df = news_df 
        top_news = pd.DataFrame()

        if 'timestamp' in latest_news_df.columns:
            latest_news_df['datetime'] = pd.to_datetime(latest_news_df['timestamp'])
            one_hour_ago = now_ist - timedelta(hours=1)
            
            # Filter news published in the last hour
            # Avoiding tz_localize for now due to complexity in data types, relying on naive comparison if possible
            news_last_hour = latest_news_df[pd.to_datetime(latest_news_df['timestamp']).dt.tz_localize(IST, errors='coerce') >= one_hour_ago]
            
            top_news = news_last_hour.sort_values(by='datetime', ascending=False).head(2)
        else:
            top_news = latest_news_df.head(2) # Fallback if no timestamp

        # Render the notification content inside the expander
        st.markdown("<h4 style='color:#1E90FF; margin-top:10px;'>🔥 Live Market Movers</h4>", unsafe_allow_html=True)
        
        col_g, col_l = st.columns(2)
        with col_g:
            st.markdown("<h5 style='color:green; margin-bottom:5px;'>Top 2 Gainers</h5>", unsafe_allow_html=True)
            st.dataframe(market_gainers.style.format({"change_pct": "{:+.2f}%"}).hide(axis="index"), use_container_width=True)
        
        with col_l:
            st.markdown("<h5 style='color:red; margin-bottom:5px;'>Top 2 Losers</h5>", unsafe_allow_html=True)
            st.dataframe(market_losers.style.format({"change_pct": "{:+.2f}%"}).hide(axis="index"), use_container_width=True)

        st.markdown("<h5 style='margin-top:20px;'>Latest News (Sample)</h5>", unsafe_allow_html=True)
        if not top_news.empty and 'title' in top_news.columns:
             for _, row in top_news.iterrows():
                sentiment_emoji = "🟢" if row.get("sentiment") == "Bullish" else "🔴" if row.get("sentiment") == "Bearish" else "⚪"
                st.markdown(f"**{sentiment_emoji} {row['title']}** - ({row.get('symbol', 'N/A')})", unsafe_allow_html=True)
        else:
            st.info("No recent news available.")
    else:
        st.info("No stock data available for notifications.")


# =====================================
# MAIN CONTENT: Gainers/Losers Metrics (Market Wide - NO HEADING)
# =====================================

st.markdown("---")

# Get ALL symbols to ensure we look at the whole market for gainers/losers
all_symbols = stocks_df["symbol"].unique().tolist() if not stocks_df.empty and "symbol" in stocks_df.columns else []
df_all_market_data = preprocess_stocks(stocks_df, all_symbols) 


if not df_all_market_data.empty:
    # Calculate top movers from the complete, unfiltered dataset
    top_gainers_all = df_all_market_data.nlargest(5, "change_pct")
    top_losers_all = df_all_market_data.nsmallest(5, "change_pct")

    col_gainer, col_loser = st.columns(2)

    with col_gainer:
        st.markdown("<h4 style='color:green;'>Top 5 Gainers</h4>", unsafe_allow_html=True)
        for _, row in top_gainers_all.iterrows():
            st.metric(
                label=row["symbol"], 
                value=f"{row['close']:.2f}", 
                delta=f"{row['change_pct']:+.2f}%"
            )

    with col_loser:
        st.markdown("<h4 style='color:red;'>Top 5 Losers</h4>", unsafe_allow_html=True)
        for _, row in top_losers_all.iterrows():
            st.metric(
                label=row["symbol"], 
                value=f"{row['close']:.2f}", 
                delta=f"{row['change_pct']:+.2f}%"
            )

else:
    st.warning("No data available to determine top market movers.")


# =====================================
# BUY/SELL PREDICTIONS (Top 5 Movers overall)
# =====================================
st.markdown("---")
st.subheader("🤖 Top 5 Movers: Latest Model Recommendations")
st.caption("Recommendations for the 5 stocks with the highest absolute market movement.")

if not buy_sell_df.empty and not df_all_market_data.empty:
    # 1. Get the top 5 movers based on ABSOLUTE change
    df_all_market_data['abs_change'] = df_all_market_data['change_pct'].abs()
    top_5_movers = df_all_market_data.nlargest(5, "abs_change")['symbol'].unique()
    
    # 2. Filter predictions for these movers
    predictions_movers = buy_sell_df[buy_sell_df["symbol"].isin(top_5_movers)]
    
    if not predictions_movers.empty:
        # Prepare data for display: ensure 'action', 'buy_pred', 'sell_pred' exist
        predictions_movers = predictions_movers.copy()
        predictions_movers['Action'] = predictions_movers.get('action', 'HOLD').str.upper()
        predictions_movers['Buy_Conf'] = predictions_movers.get('buy_pred', 0.0)
        predictions_movers['Sell_Conf'] = predictions_movers.get('sell_pred', 0.0)
        
        # Merge with market data to show current % change for context
        predictions_movers = pd.merge(
            predictions_movers, 
            df_all_market_data[['symbol', 'change_pct']], 
            on='symbol', 
            how='left'
        ).rename(columns={'change_pct': 'Market Change %'})
        
        # Sort by market change for better visualization (gainers first)
        predictions_movers = predictions_movers.sort_values(by='Market Change %', ascending=False)
        
        # Limit to 5 records as requested
        predictions_movers = predictions_movers.head(5)
        
        # Set up table headers 
        header_cols = st.columns([2, 2, 2, 2, 2])
        header_cols[0].markdown("<div class='prediction-header'>Symbol</div>", unsafe_allow_html=True)
        header_cols[1].markdown("<div class='prediction-header'>Market Change 📈</div>", unsafe_allow_html=True)
        header_cols[2].markdown("<div class='prediction-header'>Model Action 🤖</div>", unsafe_allow_html=True)
        header_cols[3].markdown("<div class='prediction-header'>Buy Confidence</div>", unsafe_allow_html=True)
        header_cols[4].markdown("<div class='prediction-header'>Sell Confidence</div>", unsafe_allow_html=True)
        
        # Iterate and display data rows
        for _, row in predictions_movers.iterrows():
            action = row['Action']
            buy_conf = row['Buy_Conf'] * 100
            sell_conf = row['Sell_Conf'] * 100
            market_change = row['Market Change %']
            
            # Action styling (Green ▲ for BUY, Red ▼ for SELL)
            if action == "BUY":
                action_text = f"<span style='color:green; font-weight:bold;'>{action} ▲</span>"
            elif action == "SELL":
                action_text = f"<span style='color:red; font-weight:bold;'>{action} ▼</span>"
            else:
                action_text = f"<span style='color:#6c757d; font-weight:bold;'>{action} ➡️</span>" 
                
            # Market change styling
            change_color = 'green' if market_change >= 0 else 'red'
            change_text = f"<span style='color:{change_color};'>{market_change:+.2f}%</span>"
            
            # Confidence formatting (2 decimal points)
            buy_conf_text = f"<span style='color:green;'>{buy_conf:.2f}%</span>"
            sell_conf_text = f"<span style='color:red;'>{sell_conf:.2f}%</span>"


            row_cols = st.columns([2, 2, 2, 2, 2])
            row_cols[0].markdown(f"**{row['symbol']}**")
            row_cols[1].markdown(change_text, unsafe_allow_html=True)
            row_cols[2].markdown(action_text, unsafe_allow_html=True)
            row_cols[3].markdown(buy_conf_text, unsafe_allow_html=True)
            row_cols[4].markdown(sell_conf_text, unsafe_allow_html=True)
            
            st.markdown("<hr style='margin-top:0px; margin-bottom: 0px;'>", unsafe_allow_html=True)
            
    else:
        st.info("No prediction data available for the current top market movers.")

else:
    st.info("Prediction or stock data is not available.")


# =====================================
# FILTERED NEWS SENTIMENT (Latest 2 records for selected symbols)
# =====================================
st.markdown("---")
st.subheader("📰 Sentiment Analysis (Latest News for Selected Stocks)")
st.caption(f"Showing the **2** most recent news articles for selected stocks: {', '.join(selected_symbols) if selected_symbols else 'None'}")

if not news_df.empty and selected_symbols and 'timestamp' in news_df.columns:
    df_filtered_news = news_df[news_df['symbol'].isin(selected_symbols)].copy()
    
    # Ensure datetime format for sorting
    df_filtered_news['datetime'] = pd.to_datetime(df_filtered_news['publish_datetime'])
    
    # Get the 2 most recent news items
    top_2_news = df_filtered_news.sort_values(by='datetime', ascending=False).head(2)
    
    if not top_2_news.empty:
        # Prepare data for display
        display_news = top_2_news[['publish_datetime', 'symbol', 'title', 'sentiment']].copy()
        
        # Style the sentiment column
        def style_sentiment(s):
            if s == 'Bullish':
                return 'background-color: #e6ffe6; color: green; font-weight: bold;'
            elif s == 'Bearish':
                return 'background-color: #ffe6e6; color: red; font-weight: bold;'
            return ''

        st.dataframe(
            display_news.style.applymap(style_sentiment, subset=['sentiment']).hide(axis="index"), 
            use_container_width=True
        )
    else:
        st.info("No recent news found for the selected symbols.")
else:
    st.info("News sentiment data is not available or symbols not selected.")


# =====================================
# UPCOMING QUARTERLY EVENTS 
# =====================================
st.markdown("---")
st.subheader("🗓️ Upcoming Corporate & Economic Events")

# Placeholder for events (Simulated data since live fetching is not possible)
events_data = {
    'Date (IST)': [(now_ist.date() + timedelta(days=2)).strftime('%Y-%m-%d'), (now_ist.date() + timedelta(days=5)).strftime('%Y-%m-%d'), (now_ist.date() + timedelta(days=10)).strftime('%Y-%m-%d')],
    'Symbol/Event': ['RELIANCE', 'TCS', 'RBI Policy Meeting'],
    'Description': ['Q2 Earnings Announcement', 'Board Meeting on Dividend', 'Interest Rate Decision'],
    'Source': ['ET/Yahoo', 'ET/Yahoo', 'RBI'],
}
events_df = pd.DataFrame(events_data)

st.dataframe(events_df.style.set_properties(**{'font-size': '10pt', 'border': '1px solid #ddd'}).hide(axis="index"), use_container_width=True)
