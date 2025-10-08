import streamlit as st
import pandas as pd
import yfinance as yf
import os
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import altair as alt

# ==========================
# DATABASE CONFIG
# ==========================
db_user = "<your_user>"
db_password = "<your_password>"
db_host = "<your_host>"
db_port = "5432"
db_name = "<your_db>"

DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
DB_CONNECTED = False
try:
    engine = create_engine(DB_URI, connect_args={'connect_timeout':5})
    with engine.connect():
        DB_CONNECTED = True
except:
    DB_CONNECTED = False

# ==========================
# CACHE PATH
# ==========================
DATA_PATH = "./cache"
os.makedirs(DATA_PATH, exist_ok=True)

# ==========================
# LOAD TABLES
# ==========================
def load_table(table_name):
    cache_file = f"{DATA_PATH}/{table_name}.csv"
    df = None
    if DB_CONNECTED:
        try:
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        except:
            pass
    if df is not None and not df.empty:
        df.to_csv(cache_file, index=False)
        return df
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file)
    return pd.DataFrame()

stocks_df = load_table("stocks")
pred_df = load_table("buy_sell_predictions")
news_df = load_table("news_sentiment")

# ==========================
# SIDEBAR CONTROLS
# ==========================
st.sidebar.header("⚙️ Controls")
symbols = sorted(stocks_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks", symbols, default=symbols[:3])
interval = st.sidebar.selectbox("Select Interval", ["5m","15m"])
if st.sidebar.button("🔄 Refresh Data"):
    st.experimental_rerun = lambda: None  # workaround
    st.experimental_rerun()  # simulate refresh

# ==========================
# TIMEZONE & TODAY
# ==========================
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist)
today_date = now_ist.date()

# ==========================
# ML MODEL RECOMMENDATIONS
# ==========================
st.subheader("💹 ML Model Recommendations")
if not pred_df.empty:
    df_pred = pred_df.head(10).copy()
    df_pred = df_pred.drop(columns=["timestamp"], errors="ignore")
    
    # Create 'Action' column with icons
    df_pred["Action"] = df_pred["action"].apply(
        lambda a: "⬆️ Buy" if str(a).upper()=="BUY" else "⬇️ Sell" if str(a).upper()=="SELL" else "⏸️ No Trade"
    )
    
    # Remove duplicate columns
    df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()]
    
    # Display columns (exclude original 'action')
    display_cols = [col for col in df_pred.columns if col not in ["action"]]

    # Highlight colors
    def highlight_action(val):
        if "Buy" in val:
            return "color: green; font-weight: bold"
        elif "Sell" in val:
            return "color: red; font-weight: bold"
        return ""
    
    st.dataframe(df_pred[display_cols].style.applymap(highlight_action, subset=["Action"]),
                 width="stretch")
else:
    st.info("No Buy/Sell predictions available.")

# ==========================
# NEWS SENTIMENT
# ==========================
st.subheader("📰 News Sentiment")
if not news_df.empty:
    chart_data = pd.DataFrame({
        "Category": ["Bullish","Bearish","Neutral"],
        "Count": [news_df[news_df["sentiment"]=="Bullish"].shape[0],
                  news_df[news_df["sentiment"]=="Bearish"].shape[0],
                  news_df[news_df["sentiment"]=="Neutral"].shape[0]]
    })
    chart = alt.Chart(chart_data).mark_bar().encode(
        x="Count",
        y=alt.Y("Category", sort="-x"),
        color="Category"
    )
    st.altair_chart(chart, width="stretch")
else:
    st.write("No sentiment data available.")

# ==========================
# LAST UPDATED TIMESTAMP
# ==========================
st.markdown(f"<p style='text-align:right; font-size:12px;'>Last Updated: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
