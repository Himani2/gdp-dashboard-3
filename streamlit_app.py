# import streamlit as st
# import pandas as pd
# import math
# from pathlib import Path

# # Set the title and favicon that appear in the Browser's tab bar.
# st.set_page_config(
#     page_title='GDP dashboard',
#     page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
# )

# # -----------------------------------------------------------------------------
# # Declare some useful functions.

# @st.cache_data
# def get_gdp_data():
#     """Grab GDP data from a CSV file.

#     This uses caching to avoid having to read the file every time. If we were
#     reading from an HTTP endpoint instead of a file, it's a good idea to set
#     a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
#     """

#     # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
#     DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
#     raw_gdp_df = pd.read_csv(DATA_FILENAME)

#     MIN_YEAR = 1960
#     MAX_YEAR = 2022

#     # The data above has columns like:
#     # - Country Name
#     # - Country Code
#     # - [Stuff I don't care about]
#     # - GDP for 1960
#     # - GDP for 1961
#     # - GDP for 1962
#     # - ...
#     # - GDP for 2022
#     #
#     # ...but I want this instead:
#     # - Country Name
#     # - Country Code
#     # - Year
#     # - GDP
#     #
#     # So let's pivot all those year-columns into two: Year and GDP
#     gdp_df = raw_gdp_df.melt(
#         ['Country Code'],
#         [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
#         'Year',
#         'GDP',
#     )

#     # Convert years from string to integers
#     gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

#     return gdp_df

# gdp_df = get_gdp_data()

# # -----------------------------------------------------------------------------
# # Draw the actual page

# # Set the title that appears at the top of the page.
# '''
# # :earth_americas: GDP dashboard

# Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
# notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
# But it's otherwise a great (and did I mention _free_?) source of data.
# '''

# # Add some spacing
# ''
# ''

# min_value = gdp_df['Year'].min()
# max_value = gdp_df['Year'].max()

# from_year, to_year = st.slider(
#     'Which years are you interested in?',
#     min_value=min_value,
#     max_value=max_value,
#     value=[min_value, max_value])

# countries = gdp_df['Country Code'].unique()

# if not len(countries):
#     st.warning("Select at least one country")

# selected_countries = st.multiselect(
#     'Which countries would you like to view?',
#     countries,
#     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# ''
# ''
# ''

# # Filter the data
# filtered_gdp_df = gdp_df[
#     (gdp_df['Country Code'].isin(selected_countries))
#     & (gdp_df['Year'] <= to_year)
#     & (from_year <= gdp_df['Year'])
# ]

# st.header('GDP over time', divider='gray')

# ''

# st.line_chart(
#     filtered_gdp_df,
#     x='Year',
#     y='GDP',
#     color='Country Code',
# )

# ''
# ''


# first_year = gdp_df[gdp_df['Year'] == from_year]
# last_year = gdp_df[gdp_df['Year'] == to_year]

# st.header(f'GDP in {to_year}', divider='gray')

# ''

# cols = st.columns(4)

# for i, country in enumerate(selected_countries):
#     col = cols[i % len(cols)]

#     with col:
#         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
#         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

#         if math.isnan(first_gdp):
#             growth = 'n/a'
#             delta_color = 'off'
#         else:
#             growth = f'{last_gdp / first_gdp:,.2f}x'
#             delta_color = 'normal'

#         st.metric(
#             label=f'{country} GDP',
#             value=f'{last_gdp:,.0f}B',
#             delta=growth,
#             delta_color=delta_color
#         )

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import psycopg2

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

