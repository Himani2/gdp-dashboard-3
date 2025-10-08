import streamlit as st
import pandas as pd
import yfinance as yf
import os
import pytz
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import feedparser
import altair as alt

# ==========================
# DATABASE CONFIG
# ==========================
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"

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
SECTOR_CSV = f"{DATA_PATH}/sectors.csv"

# ==========================
# MARKET HOLIDAYS
# ==========================
MARKET_HOLIDAYS = [
    datetime(2025,2,26).date(), datetime(2025,3,14).date(), datetime(2025,3,31).date(),
    datetime(2025,4,10).date(), datetime(2025,4,14).date(), datetime(2025,4,18).date(),
    datetime(2025,5,1).date(), datetime(2025,8,15).date(), datetime(2025,8,27).date(),
    datetime(2025,10,2).date(), datetime(2025,10,21).date(), datetime(2025,10,22).date(),
    datetime(2025,11,5).date(), datetime(2025,12,25).date()
]
HOLIDAY_NAMES = {
    datetime(2025,2,26).date(): "Mahashivratri",
    datetime(2025,3,14).date(): "Holi",
    datetime(2025,3,31).date(): "Eid-Ul-Fitr",
    datetime(2025,4,10).date(): "Mahavir Jayanti",
    datetime(2025,4,14).date(): "Ambedkar Jayanti",
    datetime(2025,4,18).date(): "Good Friday",
    datetime(2025,5,1).date(): "Maharashtra Day",
    datetime(2025,8,15).date(): "Independence Day",
    datetime(2025,8,27).date(): "Ganesh Chaturthi",
    datetime(2025,10,2).date(): "Gandhi Jayanti/Dussehra",
    datetime(2025,10,21).date(): "Diwali Laxmi Pujan",
    datetime(2025,10,22).date(): "Diwali-Balipratipada",
    datetime(2025,11,5).date(): "Guru Nanak Dev Jayanti",
    datetime(2025,12,25).date(): "Christmas",
}

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
buy_sell_df = load_table("buy_sell_predictions")
pred_df = load_table("buy_sell_predictions")
news_df = load_table("news_sentiment")

# ==========================
# FETCH SECTOR DATA
# ==========================
def fetch_sectors(symbols):
    sectors = []
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            sectors.append({"symbol": sym, "sector": info.get("sector", "Unknown")})
        except:
            sectors.append({"symbol": sym, "sector": "Unknown"})
    df = pd.DataFrame(sectors)
    df.to_csv(SECTOR_CSV, index=False)
    return df

if os.path.exists(SECTOR_CSV):
    sectors_df = pd.read_csv(SECTOR_CSV)
else:
    symbols = stocks_df["symbol"].unique().tolist()
    sectors_df = fetch_sectors(symbols)

if "sector" not in stocks_df.columns:
    stocks_df = stocks_df.merge(sectors_df, on="symbol", how="left")

# ==========================
# SIDEBAR CONTROLS
# ==========================
st.sidebar.header("⚙️ Controls")
symbols = sorted(stocks_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks", symbols, default=symbols[:3])
interval = st.sidebar.selectbox("Select Interval", ["5m","15m"])
if st.sidebar.button("🔄 Refresh Data"):
    if os.path.exists(SECTOR_CSV):
        os.remove(SECTOR_CSV)
    sectors_df = fetch_sectors(symbols)
    st.experimental_rerun = lambda: None  # workaround
    st.experimental_rerun()  # simulate refresh

# ==========================
# TIMEZONE & TODAY
# ==========================
ist = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(ist)
today_date = now_ist.date()

# ==========================
# METRICS CARDS
# ==========================
# Adjust Friday data if weekend
last_trade_date = today_date
if today_date.weekday() >=5:
    last_trade_date -= timedelta(days=(today_date.weekday()-4))

stocks_today = stocks_df[stocks_df["timestamp"].str.contains(str(last_trade_date))]
if not stocks_today.empty:
    stocks_today["change_pct"] = (stocks_today["close"]-stocks_today["prev_close"])/stocks_today["prev_close"]*100
    top_gainer = stocks_today.loc[stocks_today["change_pct"].idxmax()]
    top_loser = stocks_today.loc[stocks_today["change_pct"].idxmin()]
    most_active = stocks_today.loc[stocks_today["volume"].idxmax()]
    least_active = stocks_today.loc[stocks_today["volume"].idxmin()]
    
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Top Gainer", f"{top_gainer['symbol']}", f"{top_gainer['change_pct']:+.2f}%")
    col2.metric("Top Loser", f"{top_loser['symbol']}", f"{top_loser['change_pct']:+.2f}%")
    col3.metric("Most Active", f"{most_active['symbol']}", f"{most_active['volume']}")
    col4.metric("Least Active", f"{least_active['symbol']}", f"{least_active['volume']}")

st.markdown(f"<p style='text-align:right; font-size:12px;'>Last Updated: {now_ist.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

# ==========================
# ML MODEL RECOMMENDATIONS
# ==========================
st.subheader("💹 ML Model Recommendations")
if not pred_df.empty:
    df_ml = pred_df.head(10).copy()
    df_ml = df_ml.drop(columns=["timestamp"], errors="ignore")
    df_ml["action"] = df_ml["action"].apply(lambda a: "⬆️ Buy" if a=="BUY" else "⬇️ Sell" if a=="SELL" else "⏸️ No Trade")
    def highlight_action(val):
        if "Buy" in val:
            return "color: green"
        elif "Sell" in val:
            return "color: red"
        return ""
    st.dataframe(df_ml.style.applymap(highlight_action, subset=["action"]), use_container_width=True)

# ==========================
# NEWS SENTIMENT
# ==========================
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
    st.altair_chart(chart, use_container_width=True)
