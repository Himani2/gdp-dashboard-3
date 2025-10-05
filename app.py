import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")

st.title("News Sentiment Analysis Dashboard")

# 1️⃣ Database connection
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"

@st.cache_resource
def get_engine():
    return create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

engine = get_engine()

# 2️⃣ Load data
@st.cache_data(show_spinner=False)
def load_data():
    with st.spinner("Loading data from database..."):
        df = pd.read_sql("SELECT * FROM news_sentiment", con=engine)
        # Preprocess
        df['publish_datetime'] = pd.to_datetime(df['publish_datetime'])
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        df['sentiment_score'] = df['sentiment'].map(sentiment_map)
        df['day'] = df['publish_datetime'].dt.date
        return df

df = load_data()

if not df.empty:
    # 3️⃣ Visualization 1: Average Sentiment Trend Over Time
    st.subheader("Average News Sentiment Trend Over Time")
    sentiment_trend = df.groupby('day')['sentiment_score'].mean()

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(sentiment_trend.index, sentiment_trend.values, marker='o', color='blue', linewidth=2)
    ax1.fill_between(sentiment_trend.index, sentiment_trend.values, color='#aec7e8', alpha=0.3)
    ax1.set_title("Average Sentiment Trend Over Time", fontsize=14)
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Sentiment Score", fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yticks([-1, 0, 1], ['Bearish', 'Neutral', 'Bullish'])
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig1)

    # 4️⃣ Visualization 2: Daily Sentiment Counts (Stacked Area Chart)
    st.subheader("Daily Sentiment Counts Over Time")
    daily_counts = df.groupby('day')['sentiment'].value_counts().unstack().fillna(0)

    for col in ['positive','negative','neutral']:
        if col not in daily_counts.columns:
            daily_counts[col] = 0
    daily_counts = daily_counts[['positive','negative','neutral']]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    daily_counts.plot(kind='area', stacked=True, ax=ax2, alpha=0.7, color=['green','red','gray'])
    ax2.set_title("Daily Sentiment Counts Over Time", fontsize=14)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Number of Articles", fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title="Sentiment")
    plt.tight_layout()
    st.pyplot(fig2)

    # 5️⃣ Add placeholders for other visualizations
    st.markdown("---")
    st.subheader("Sentiment Trend for Top 5 Stocks")
    st.write("*(Code for this visualization can be added here)*")
    # Add code for plotting sentiment trend for top 5 stocks

    st.markdown("---")
    st.subheader("Overall Sentiment for Top 5 Stocks with Representative News")
    st.write("*(Code for this visualization can be added here)*")
    # Add code for plotting overall sentiment for top 5 stocks with news headlines

else:
    st.warning("No data available in the 'news_sentiment' table.")
