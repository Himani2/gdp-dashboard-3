import os
import streamlit as st
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import feedparser
import datetime
from datetime import timedelta
from typing import List, Dict, Any, Optional

# =======================================
# CONFIGURATION AND SETUP
# =======================================

# --- Database Placeholders ---
db_user = "postgres"
# Database credentials are now hardcoded to the default fallback values 
# as requested to simplify the setup.
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"

DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

DB_CONNECTED = False
try:
    # Use a timeout context manager or connection test here in a real application
    engine = create_engine(DB_URI, connect_args={'connect_timeout': 5})
    with engine.connect():
        DB_CONNECTED = True
except Exception as e:
    # st.error(f"Database connection failed: {e}") # Uncomment for debugging
    DB_CONNECTED = False
    pass 


# --- Market Holidays for NSE/BSE (2025 Equity Segment) ---
# Used to check if today is a non-trading day.
# Source: NSE/BSE holiday calendar 2025.
NSE_HOLIDAYS_2025 = [
    datetime.date(2025, 2, 26),  # Mahashivratri (Wednesday)
    datetime.date(2025, 3, 14),  # Holi (Friday)
    datetime.date(2025, 3, 31),  # Eid-Ul-Fitr (Monday)
    datetime.date(2025, 4, 10),  # Shri Mahavir Jayanti (Thursday)
    datetime.date(2025, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti (Monday)
    datetime.date(2025, 4, 18),  # Good Friday (Friday)
    datetime.date(2025, 5, 1),   # Maharashtra Day (Thursday)
    datetime.date(2025, 8, 15),  # Independence Day (Friday)
    datetime.date(2025, 8, 27),  # Ganesh Chaturthi (Wednesday)
    datetime.date(2025, 10, 2),  # Mahatma Gandhi Jayanti/Dussehra (Thursday)
    datetime.date(2025, 10, 21), # Diwali Laxmi Pujan (Muhurat Trading only) (Tuesday)
    datetime.date(2025, 10, 22), # Diwali-Balipratipada (Wednesday)
    datetime.date(2025, 11, 5),  # Prakash Gurpurb Sri Guru Nanak Dev (Wednesday)
    datetime.date(2025, 12, 25), # Christmas (Thursday)
]

# =======================================
# UTILITY FUNCTIONS
# =======================================

@st.cache_data(ttl=3600)
def is_trading_day(check_date: datetime.date) -> bool:
    """Checks if a given date is a trading day (Monday-Friday, non-holiday)."""
    # Check for weekends (Monday=0, Sunday=6)
    if check_date.weekday() >= 5: # Saturday or Sunday
        return False
    # Check for declared holidays
    if check_date in NSE_HOLIDAYS_2025:
        return False
    return True

@st.cache_data(ttl=3600)
def get_latest_trading_date(start_date: datetime.date) -> datetime.date:
    """Finds the most recent previous trading day."""
    current_date = start_date
    while not is_trading_day(current_date):
        current_date -= timedelta(days=1)
    return current_date

@st.cache_data(ttl=3600)
def load_yfinance_fallback(ticker: str, periods: str) -> Optional[pd.DataFrame]:
    """
    Loads data from yfinance. If today is a non-trading day, it loads the data 
    from the previous trading day to ensure fresh market data is displayed.
    """
    try:
        # Determine the correct date range to fetch
        today = datetime.date.today()
        latest_date = get_latest_trading_date(today)
        
        # Adjust the 'end' date for yfinance to ensure we capture the latest trading day's data
        # We use tomorrow's date for 'end' to include the full 'latest_date' data.
        end_date = latest_date + timedelta(days=1)
        start_date = latest_date - timedelta(days=365) # Last 1 year for history
        
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if data.empty:
            st.warning(f"No data fetched for {ticker} on {latest_date}. Check ticker or holiday status.")
            return None

        # Filter to ensure we only show data up to the latest trading day
        data = data[data.index.date <= latest_date]
        
        return data

    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for {ticker}: {e}")
        return None

# =======================================
# DATA LOADING FUNCTIONS
# =======================================

@st.cache_data(ttl=3600)
def load_market_data(ticker: str) -> Optional[pd.DataFrame]:
    """Loads market data, preferring DB, falling back to yfinance."""
    if DB_CONNECTED:
        try:
            # Assumes 'stocks' table has 'date' and 'close' columns
            df = pd.read_sql(f"SELECT date, close FROM stocks WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT 365", engine)
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.set_index('date').sort_index()
            return df
        except Exception as e:
            st.warning(f"DB Load failed for {ticker}. Falling back to yfinance. Error: {e}")
            return load_yfinance_fallback(ticker, "1y")
    else:
        return load_yfinance_fallback(ticker, "1y")


@st.cache_data(ttl=3600)
def load_recommendations() -> pd.DataFrame:
    """Loads latest predictions, preferring DB, falling back to dummy data."""
    if DB_CONNECTED:
        try:
            # Assumes 'buy_sell_recommendation' table is structured appropriately
            df = pd.read_sql("SELECT * FROM buy_sell_recommendation ORDER BY date DESC LIMIT 5", engine)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception:
            st.warning("DB Load failed for recommendations. Returning empty data.")
    
    # Fallback: Return an empty DataFrame if DB connection fails (Removed Dummy Data)
    return pd.DataFrame(columns=['date', 'ticker', 'recommendation', 'confidence'])


@st.cache_data(ttl=3600)
def load_news_fallback() -> pd.DataFrame:
    """Loads latest economic news from an RSS feed."""
    try:
        feed_url = "https://economictimes.indiatimes.com/rssfeedcode/2552695.cms" # Indian Markets/Economy RSS
        feed = feedparser.parse(feed_url)
        
        news_data = []
        for entry in feed.entries[:10]:
            news_data.append({
                'title': entry.title,
                'summary': entry.summary,
                'link': entry.link,
                'published': pd.to_datetime(entry.published).tz_convert(None) if hasattr(entry, 'published') else datetime.datetime.now()
            })
        return pd.DataFrame(news_data)
    except Exception:
        # Emergency fallback: return an empty DataFrame if RSS fails (Removed Dummy Data)
        st.warning("RSS Feed load failed. Returning empty news data.")
        return pd.DataFrame(columns=['title', 'summary', 'link', 'published'])

@st.cache_data(ttl=3600)
def load_news_data() -> pd.DataFrame:
    """Loads news data, preferring DB, falling back to RSS."""
    if DB_CONNECTED:
        try:
            # Assumes 'news_sentiment' table is structured appropriately
            df = pd.read_sql("SELECT title, sentiment, score, published_at FROM news_sentiment ORDER BY published_at DESC LIMIT 10", engine)
            df.rename(columns={'published_at': 'published'}, inplace=True)
            return df
        except Exception:
            st.warning("DB Load failed for news. Using RSS fallback.")
    
    # If DB fails or not connected, use the RSS fallback
    return load_news_fallback()

# =======================================
# VISUALIZATION AND DISPLAY FUNCTIONS
# =======================================

def display_dashboard(market_data: pd.DataFrame, recommendations: pd.DataFrame, news_data: pd.DataFrame, trading_date: datetime.date):
    """Renders the main Streamlit dashboard layout."""
    
    st.title("🇮🇳 GenAI Indian Market Dashboard")
    st.caption(f"Market Data as of: **{trading_date.strftime('%Y-%m-%d')}**")
    
    # --- Holiday Card ---
    col1, col2, col3 = st.columns(3)
    
    upcoming_holidays = sorted([d for d in NSE_HOLIDAYS_2025 if d > datetime.date.today()])
    
    if upcoming_holidays:
        next_holiday_date = upcoming_holidays[0]
        next_holiday_name = get_holiday_name(next_holiday_date)
        days_away = (next_holiday_date - datetime.date.today()).days
        
        col1.metric(
            label="Next Market Holiday", 
            value=next_holiday_name, 
            delta=f"{days_away} days away ({next_holiday_date.strftime('%b %d')})"
        )
    else:
        col1.metric(label="Next Market Holiday", value="Calendar Clear", delta="No 2025 holidays remaining")

    # --- Market Metrics ---
    # FIX: Use .item() to extract the scalar value from the Pandas Series element
    latest_close = market_data['Close'].iloc[-1].item()
    previous_close = market_data['Close'].iloc[-2].item()
    change = latest_close - previous_close
    percent_change = (change / previous_close) * 100

    col2.metric(
        label="Nifty 50 Close", 
        value=f"{latest_close:,.2f}", 
        delta=f"{change:+.2f} ({percent_change:+.2f}%)"
    )

    # Ensure recommendations is not empty before attempting to filter
    buy_count = 0
    if not recommendations.empty:
        buy_count = recommendations[recommendations['recommendation'] == 'BUY'].shape[0]

    col3.metric(
        label="Recommendation Count (Buy)", 
        value=buy_count,
        delta="Based on latest model run"
    )

    st.markdown("---")

    # --- Chart and Recommendations ---
    st.header("Nifty 50 Trend")
    st.line_chart(market_data['Close'])

    st.header("Latest AI Recommendations")
    # This correctly displays an empty table if recommendations is an empty DataFrame
    st.dataframe(recommendations) 

    # --- News and Sentiment ---
    st.header("Market News & Sentiment")
    
    if 'sentiment' in news_data.columns and not news_data.empty:
        # Display aggregated sentiment if DB data is used and available
        st.subheader("Aggregated Sentiment")
        sentiment_counts = news_data['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
    elif 'sentiment' in news_data.columns and news_data.empty:
        st.info("No recent news or sentiment data available.")

    st.subheader("Latest News Articles")
    
    # Only iterate if news_data is not empty
    if not news_data.empty:
        for index, row in news_data.head(5).iterrows():
            title = row['title']
            summary = row.get('summary', 'No summary available.')
            link = row.get('link', '#')
            published = row.get('published', 'N/A')
            sentiment = row.get('sentiment', 'N/A')
            
            st.markdown(f"""
                <div style="padding: 10px; margin-bottom: 10px; border-radius: 8px; background-color: #0d1117; border: 1px solid #1f2937;">
                    <p style="font-size: 1.1em; font-weight: bold; margin: 0;">
                        <a href="{link}" target="_blank" style="color: #6366f1; text-decoration: none;">{title}</a>
                    </p>
                    <p style="font-size: 0.9em; margin-top: 5px; margin-bottom: 5px; color: #9ca3af;">
                        {summary[:100]}...
                    </p>
                    <p style="font-size: 0.75em; margin: 0; color: #4b5563;">
                        Published: {published} | Sentiment: <b>{sentiment}</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news articles to display.")
        
    st.markdown("---")
    
    # --- Holiday List Card ---
    st.header("Indian Stock Market Holidays (2025)")
    holiday_details = []
    for date in sorted(NSE_HOLIDAYS_2025):
        name = get_holiday_name(date)
        # Create HTML list item for each holiday
        holiday_details.append(f'<li style="margin-bottom: 5px; color: #d1d5db;">- **{date.strftime("%b %d, %Y")}**: {name}</li>')
    
    # FIX: Join the list items outside the f-string to avoid the backslash error
    list_html_content = "\n".join(holiday_details)
    
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 12px; background-color: #1a202c; border: 2px solid #6366f1;">
            <p style="font-weight: bold; color: #fff;">Equity and Derivative Markets will be closed on these days:</p>
            <ul style="list-style-type: none; padding-left: 0;">
                {list_html_content}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
def get_holiday_name(date: datetime.date) -> str:
    """Helper function to map date to holiday name."""
    mapping = {
        datetime.date(2025, 2, 26): "Mahashivratri",
        datetime.date(2025, 3, 14): "Holi",
        datetime.date(2025, 3, 31): "Eid-Ul-Fitr",
        datetime.date(2025, 4, 10): "Mahavir Jayanti",
        datetime.date(2025, 4, 14): "Ambedkar Jayanti",
        datetime.date(2025, 4, 18): "Good Friday",
        datetime.date(2025, 5, 1): "Maharashtra Day",
        datetime.date(2025, 8, 15): "Independence Day",
        datetime.date(2025, 8, 27): "Ganesh Chaturthi",
        datetime.date(2025, 10, 2): "Gandhi Jayanti/Dussehra",
        datetime.date(2025, 10, 21): "Diwali Laxmi Pujan",
        datetime.date(2025, 10, 22): "Diwali-Balipratipada",
        datetime.date(2025, 11, 5): "Guru Nanak Dev Jayanti",
        datetime.date(2025, 12, 25): "Christmas",
    }
    return mapping.get(date, "Holiday")


# =======================================
# MAIN EXECUTION
# =======================================
def main():
    """Main function to run the dashboard."""
    # Check if the current time is a non-trading period (e.g., late at night, weekend, or holiday)
    # Note: Stock markets usually close at 3:30 PM IST. We check for a new trading day.
    today = datetime.date.today()
    latest_trading_date = get_latest_trading_date(today)
    
    # Load all required data based on the latest trading day
    nifty_data = load_market_data("^NSEI") 
    recommendations = load_recommendations()
    news_data = load_news_data()

    if nifty_data is not None and not nifty_data.empty:
        display_dashboard(nifty_data, recommendations, news_data, latest_trading_date)
    else:
        st.error("Could not load Nifty 50 data.")

if __name__ == "__main__":
    main()
