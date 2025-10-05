# # import streamlit as st
# # import pandas as pd
# # import math
# # from pathlib import Path

# # # Set the title and favicon that appear in the Browser's tab bar.
# # st.set_page_config(
# #     page_title='GDP dashboard',
# #     page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
# # )

# # # -----------------------------------------------------------------------------
# # # Declare some useful functions.

# # @st.cache_data
# # def get_gdp_data():
# #     """Grab GDP data from a CSV file.

# #     This uses caching to avoid having to read the file every time. If we were
# #     reading from an HTTP endpoint instead of a file, it's a good idea to set
# #     a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
# #     """

# #     # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
# #     DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
# #     raw_gdp_df = pd.read_csv(DATA_FILENAME)

# #     MIN_YEAR = 1960
# #     MAX_YEAR = 2022

# #     # The data above has columns like:
# #     # - Country Name
# #     # - Country Code
# #     # - [Stuff I don't care about]
# #     # - GDP for 1960
# #     # - GDP for 1961
# #     # - GDP for 1962
# #     # - ...
# #     # - GDP for 2022
# #     #
# #     # ...but I want this instead:
# #     # - Country Name
# #     # - Country Code
# #     # - Year
# #     # - GDP
# #     #
# #     # So let's pivot all those year-columns into two: Year and GDP
# #     gdp_df = raw_gdp_df.melt(
# #         ['Country Code'],
# #         [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
# #         'Year',
# #         'GDP',
# #     )

# #     # Convert years from string to integers
# #     gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

# #     return gdp_df

# # gdp_df = get_gdp_data()

# # # -----------------------------------------------------------------------------
# # # Draw the actual page

# # # Set the title that appears at the top of the page.
# # '''
# # # :earth_americas: GDP dashboard

# # Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
# # notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
# # But it's otherwise a great (and did I mention _free_?) source of data.
# # '''

# # # Add some spacing
# # ''
# # ''

# # min_value = gdp_df['Year'].min()
# # max_value = gdp_df['Year'].max()

# # from_year, to_year = st.slider(
# #     'Which years are you interested in?',
# #     min_value=min_value,
# #     max_value=max_value,
# #     value=[min_value, max_value])

# # countries = gdp_df['Country Code'].unique()

# # if not len(countries):
# #     st.warning("Select at least one country")

# # selected_countries = st.multiselect(
# #     'Which countries would you like to view?',
# #     countries,
# #     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# # ''
# # ''
# # ''

# # # Filter the data
# # filtered_gdp_df = gdp_df[
# #     (gdp_df['Country Code'].isin(selected_countries))
# #     & (gdp_df['Year'] <= to_year)
# #     & (from_year <= gdp_df['Year'])
# # ]

# # st.header('GDP over time', divider='gray')

# # ''

# # st.line_chart(
# #     filtered_gdp_df,
# #     x='Year',
# #     y='GDP',
# #     color='Country Code',
# # )

# # ''
# # ''


# # first_year = gdp_df[gdp_df['Year'] == from_year]
# # last_year = gdp_df[gdp_df['Year'] == to_year]

# # st.header(f'GDP in {to_year}', divider='gray')

# # ''

# # cols = st.columns(4)

# # for i, country in enumerate(selected_countries):
# #     col = cols[i % len(cols)]

# #     with col:
# #         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
# #         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

# #         if math.isnan(first_gdp):
# #             growth = 'n/a'
# #             delta_color = 'off'
# #         else:
# #             growth = f'{last_gdp / first_gdp:,.2f}x'
# #             delta_color = 'normal'

# #         st.metric(
# #             label=f'{country} GDP',
# #             value=f'{last_gdp:,.0f}B',
# #             delta=growth,
# #             delta_color=delta_color
# #         )


##-----------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# from sqlalchemy import create_engine
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import psycopg2

# st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")

# st.title("News Sentiment Analysis Dashboard")

# # 1️⃣ Database connection
# db_user = "postgres"
# db_password = "oX7IDNsZF1OrTOzS75Ek"
# db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
# db_port = "5432"  # default PostgreSQL port
# db_name = "capstone_project"

# @st.cache_resource
# def get_engine():
#     return create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# engine = get_engine()

# # 2️⃣ Load data
# @st.cache_data(show_spinner=False)
# def load_data():
#     with st.spinner("Loading data from database..."):
#         df = pd.read_sql("SELECT * FROM news_sentiment", con=engine)
#         # Preprocess
#         df['publish_datetime'] = pd.to_datetime(df['publish_datetime'])
#         sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
#         df['sentiment_score'] = df['sentiment'].map(sentiment_map)
#         df['day'] = df['publish_datetime'].dt.date
#         return df

# df = load_data()

# if not df.empty:
#     # 3️⃣ Visualization 1: Average Sentiment Trend Over Time
#     st.subheader("Average News Sentiment Trend Over Time")
#     sentiment_trend = df.groupby('day')['sentiment_score'].mean()

#     fig1, ax1 = plt.subplots(figsize=(10, 5))
#     ax1.plot(sentiment_trend.index, sentiment_trend.values, marker='o', color='blue', linewidth=2)
#     ax1.fill_between(sentiment_trend.index, sentiment_trend.values, color='#aec7e8', alpha=0.3)
#     ax1.set_title("Average Sentiment Trend Over Time", fontsize=14)
#     ax1.set_xlabel("Date", fontsize=10)
#     ax1.set_ylabel("Sentiment Score", fontsize=10)
#     ax1.tick_params(axis='x', rotation=45)
#     ax1.set_yticks([-1, 0, 1], ['Bearish', 'Neutral', 'Bullish'])
#     ax1.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     st.pyplot(fig1)

#     # 4️⃣ Visualization 2: Daily Sentiment Counts (Stacked Area Chart)
#     st.subheader("Daily Sentiment Counts Over Time")
#     daily_counts = df.groupby('day')['sentiment'].value_counts().unstack().fillna(0)

#     for col in ['positive','negative','neutral']:
#         if col not in daily_counts.columns:
#             daily_counts[col] = 0
#     daily_counts = daily_counts[['positive','negative','neutral']]

#     fig2, ax2 = plt.subplots(figsize=(10, 5))
#     daily_counts.plot(kind='area', stacked=True, ax=ax2, alpha=0.7, color=['green','red','gray'])
#     ax2.set_title("Daily Sentiment Counts Over Time", fontsize=14)
#     ax2.set_xlabel("Date", fontsize=10)
#     ax2.set_ylabel("Number of Articles", fontsize=10)
#     ax2.tick_params(axis='x', rotation=45)
#     ax2.legend(title="Sentiment")
#     plt.tight_layout()
#     st.pyplot(fig2)

#     # 5️⃣ Add placeholders for other visualizations
#     st.markdown("---")
#     st.subheader("Sentiment Trend for Top 5 Stocks")
#     st.write("*(Code for this visualization can be added here)*")
#     # Add code for plotting sentiment trend for top 5 stocks

#     st.markdown("---")
#     st.subheader("Overall Sentiment for Top 5 Stocks with Representative News")
#     st.write("*(Code for this visualization can be added here)*")
#     # Add code for plotting overall sentiment for top 5 stocks with news s

# else:
#     st.warning("No data available in the 'news_sentiment' table.")

#-------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

from sqlalchemy import create_engine
from datetime import datetime, timedelta
# import numpy as np
import plotly.express as px
import os



# import random

# ------------------------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Indian Stock Monitor", page_icon="📈", layout="wide")
# # 1️⃣ Database connection
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"


DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)
CACHE_TTL = "4h"

st.title("📊 Indian Stock Intelligence Dashboard")
st.caption("Live + Cached data | Auto-fallback | Weekend-aware")

# ------------------------------------------------------------------------------
# UTILITY: unified loader with DB → fallback → CSV cache
# ------------------------------------------------------------------------------

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
    # if df is None or df.empty:
    #     df = fallback_data(table_name)
    #     st.info(f"ℹ️ Using fallback data for {table_name}")
    if table_name == 'stocks' and df is not None and not df.empty:
        return df
    elif df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
        return df
    if os.path.exists(cache_file):
        st.info(f"📁 Using cached {table_name}.csv")
        return pd.read_csv(cache_file)
    st.error(f"❌ No data for {table_name}")
    return pd.DataFrame()

# ------------------------------------------------------------------------------
# FALLBACK DATA (Yahoo Finance + synthetic)
# ------------------------------------------------------------------------------

# def fallback_data(table_name: str):
#     if table_name == "stocks":
#         symbols = ["TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK"]
#         data = []
#         for sym in symbols:
#             ticker = yf.Ticker(sym + ".NS")
#             hist = ticker.history(period="5d")
#             hist["Symbol"] = sym
#             data.append(hist.reset_index())
#         df = pd.concat(data)
#         df["Date"] = pd.to_datetime(df["Date"])
#         return df

#     elif table_name == "news_sentiment":
#         news = [
#             {"stock": "TCS", "headline": "TCS profits surge 5%", "sentiment": "bullish", "timestamp": datetime.now()},
#             {"stock": "RELIANCE", "headline": "Reliance faces margin pressure", "sentiment": "bearish", "timestamp": datetime.now()},
#             {"stock": "INFY", "headline": "Infosys announces new AI partnership", "sentiment": "bullish", "timestamp": datetime.now()},
#         ]
#         return pd.DataFrame(news)

#     elif table_name == "buy_sell_predictions":
#         data = {
#             "symbol": ["TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK"],
#             "prediction": ["Buy", "Buy", "Sell", "Hold", "Buy"],
#             "confidence": [0.91, 0.87, 0.65, 0.72, 0.89],
#             "updated_at": [datetime.now()] * 5
#         }
#         return pd.DataFrame(data)

#     return pd.DataFrame()

# ------------------------------------------------------------------------------
# LOAD ALL TABLES
# ------------------------------------------------------------------------------

stocks_df = load_or_fetch("stocks")
news_df = load_or_fetch("news_sentiment")
pred_df = load_or_fetch("buy_sell_predictions")

# ------------------------------------------------------------------------------
# MARKET TIME LOGIC: show Friday data on weekends
# ------------------------------------------------------------------------------

today = datetime.now().date()
weekday = today.weekday()  # 0=Mon ... 6=Sun
if weekday >= 5:  # Sat/Sun
    last_friday = today - timedelta(days=weekday - 4)
    st.warning(f"Market closed 🛑 Showing data from Friday ({last_friday})")
    stocks_df["timestamp"] = pd.to_datetime(stocks_df["timestamp"])
    stocks_df = stocks_df[stocks_df["timestamp"] <= pd.Timestamp(last_friday)]

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------

st.sidebar.header("⚙️ Controls")
all_symbols = sorted(stocks_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks", all_symbols, default=all_symbols[:3])
refresh = st.sidebar.button("🔄 Refresh Data")

if refresh:
    st.cache_data.clear()
    st.rerun()

#threshold = st.sidebar.slider("Alert Threshold (%)", 1, 10, 3)

# ------------------------------------------------------------------------------
# PRICE CHARTS
# ------------------------------------------------------------------------------

st.subheader("📈 Price Trend")
if not stocks_df.empty:
    fig = px.line(stocks_df[stocks_df["symbol"].isin(selected_symbols)],
                  x="timestamp", y="close", color="symbol",
                  title="Stock Closing Prices")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TOP GAINERS / LOSERS
# ------------------------------------------------------------------------------

latest_df = stocks_df.groupby("symbol").last().reset_index()
latest_df["Change%"] = ((latest_df["close"] - latest_df["open"]) / latest_df["open"]) * 100

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🟢 Top Gainers")
    st.dataframe(latest_df.sort_values("Change%", ascending=False).head(5)[["symbol", "Change%"]])
with col2:
    st.markdown("### 🔴 Top Losers")
    st.dataframe(latest_df.sort_values("Change%").head(5)[["symbol", "Change%"]])



# ------------------------------------------------------------------------------
# BUY / SELL PREDICTIONS
# ------------------------------------------------------------------------------

st.subheader("💹 Model Recommendations")
st.dataframe(pred_df[["symbol", "buy_pred", "sell_pred", "action"]])


#-----------------------------------------------------
##------------------------------------------------------------------------
##NEWS HEADLINES______
############################################################################
#----------
import pandas as pd
from sqlalchemy import create_engine
import streamlit as st

st.subheader("📰 Latest News & Sentiment")

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
    if val == "Bullish":
        return "background-color: #00FF00"
    elif val == "Bearish":
        return "background-color: #FF6666"
    else:
        return "background-color: #FFFF99"

# Fetch stock-specific news
if selected_symbols:
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
    # Map sentiment and hide stock_date
    news_df["Sentiment"] = news_df["sentiment"].apply(map_sentiment)
    news_df_display = news_df[["symbol", "title", "Sentiment"]]

    # Display color-coded table
    st.dataframe(news_df_display.style.map(color_sentiment, subset=["Sentiment"]))


#------------------------------------------


# # ------------------------------------------------------------------------------
# # ALERT POP-UPS
# # ------------------------------------------------------------------------------

# for _, row in latest_df.iterrows():
#     if abs(row["Change%"]) >= threshold:
#         emoji = "📈" if row["Change%"] > 0 else "📉"
#         st.toast(f"{emoji} {row['symbol']} moved {row['Change%']:.2f}% today!", icon="⚡")

# # ------------------------------------------------------------------------------
# # CHATBOT (basic rule-based)
# # ------------------------------------------------------------------------------

# st.subheader("💬 Stock Chatbot")
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



#-----------------------------------------------

# -----------------------
# 
