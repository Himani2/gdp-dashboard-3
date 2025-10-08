import os
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta

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

    # Price Trend
    st.subheader("📈 Price Trend")
    if not stocks_df.empty:
        fig = px.line(
            stocks_df[stocks_df["symbol"].isin(selected_symbols)],
            x="timestamp", y="close", color="symbol",
            title="Stock Closing Prices"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top Gainers & Losers
    latest_df = stocks_df.groupby("symbol").last().reset_index()
    latest_df["price_change"] = latest_df["close"] - latest_df["open"]

    def price_arrow_text(val):
        if val > 0: return f"↑ {val:.2f}"
        elif val < 0: return f"↓ {val:.2f}"
        return f"{val:.2f}"

    def color_change_cell(val):
        if isinstance(val, str):
            if val.startswith('↑'): return 'color: green; font-weight: bold;'
            elif val.startswith('↓'): return 'color: red; font-weight: bold;'
        return 'color: black;'

    top_gainers = latest_df.sort_values("price_change", ascending=False).head(5).copy()
    top_losers = latest_df.sort_values("price_change").head(5).copy()
    top_gainers["Change"] = top_gainers["price_change"].apply(price_arrow_text)
    top_losers["Change"] = top_losers["price_change"].apply(price_arrow_text)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Top 5 Gainers")
        st.dataframe(
            top_gainers[["symbol", "close", "Change"]].style.format({'close':'{:.2f}'}).applymap(color_change_cell, subset=['Change']),
            hide_index=True
        )
    with col2:
        st.markdown("### 📉 Top 5 Losers")
        st.dataframe(
            top_losers[["symbol", "close", "Change"]].style.format({'close':'{:.2f}'}).applymap(color_change_cell, subset=['Change']),
            hide_index=True
        )

    # Buy/Sell Recommendations Table
    st.subheader("💹 Buy/Sell Recommendations")
    def color_action_cell(val):
        val_str = str(val).lower()
        if "buy" in val_str: return 'color: green; font-weight: bold;'
        elif "sell" in val_str: return 'color: red; font-weight: bold;'
        elif "no trade" in val_str: return 'color: orange; font-weight: bold;'
        return 'color: black;'

    def add_arrow_to_action(action):
        action_str = str(action).lower()
        if action_str == "buy": return "⬆️ Buy"
        elif action_str == "sell": return "⬇️ Sell"
        elif action_str == "no trade": return "⏸️ No Trade"
        return action

    for col in ["buy_pred","sell_pred","price","target_price","stop_loss"]:
        if col in pred_df.columns: pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
    
    display_df = pred_df.rename(columns={
        "symbol":"Symbol",
        "action":"Action 🚦",
        "price":"Price ₹",
        "target_price":"Target 🎯",
        "stop_loss":"Stop Loss 🛑",
        "buy_pred":"Buy % ⬆️",
        "sell_pred":"Sell % ⬇️"
    }).copy()
    display_df["Action 🚦"] = display_df["Action 🚦"].apply(add_arrow_to_action)

    display_cols = ["Symbol","Action 🚦","Price ₹","Target 🎯","Stop Loss 🛑","Buy % ⬆️","Sell % ⬇️"]
    display_df = display_df[[col for col in display_cols if col in display_df.columns]]

    st.dataframe(
        display_df.style.applymap(color_action_cell, subset=["Action 🚦"])
                  .applymap(lambda x: 'color: green; font-weight: bold;', subset=["Buy % ⬆️"])
                  .applymap(lambda x: 'color: red; font-weight: bold;', subset=["Sell % ⬇️"])
                  .format({"Price ₹":"₹ {:.2f}","Target 🎯":"₹ {:.2f}","Stop Loss 🛑":"₹ {:.2f}","Buy % ⬆️":"{:.2f}","Sell % ⬇️":"{:.2f}"}),
        hide_index=True, use_container_width=True
    )

    # News & Sentiment
    st.subheader("📰 Latest News & Sentiment")
    engine = create_engine(DB_URI)

    def fetch_news(symbols=None, top_n=5):
        if symbols is None or len(symbols)==0:
            query = f"""
            SELECT n.symbol,n.title,n.sentiment,MAX(s.timestamp) AS stock_date
            FROM news_sentiment n JOIN stocks s ON n.symbol=s.symbol
            GROUP BY n.symbol,n.title,n.sentiment
            ORDER BY stock_date DESC
            LIMIT 3;"""
        else:
            symbols_list = ",".join([f"'{s.upper()}'" for s in symbols])
            query = f"""
            WITH ranked_news AS (
                SELECT s.symbol,n.title,n.sentiment,s.timestamp AS stock_date,
                ROW_NUMBER() OVER (PARTITION BY s.symbol ORDER BY s.timestamp DESC) AS rn
                FROM stocks s JOIN news_sentiment n ON s.symbol=n.symbol
                WHERE s.symbol IN ({symbols_list})
            )
            SELECT symbol,title,sentiment,stock_date
            FROM ranked_news WHERE rn<={top_n} ORDER BY stock_date DESC;
            """
        return pd.read_sql(query, engine)

    def map_sentiment(sent):
        sent = str(sent).lower()
        if sent in ["positive","bullish"]: return "Bullish"
        elif sent in ["negative","bearish"]: return "Bearish"
        return "Neutral"

    def color_sentiment(val):
        if val=="Bullish": return "color: green; font-weight: bold;"
        elif val=="Bearish": return "color: red; font-weight: bold;"
        return "color: black;"

    if 'selected_symbols' in locals() and selected_symbols:
        stock_news_df = fetch_news(selected_symbols, top_n=5)
    else:
        stock_news_df = pd.DataFrame(columns=["symbol","title","sentiment","stock_date"])

    general_news_df = fetch_news(top_n=3)
    news_df = pd.concat([stock_news_df, general_news_df], ignore_index=True)
    if not news_df.empty:
        news_df["Sentiment"] = news_df["sentiment"].apply(map_sentiment)
        news_df_display = news_df.rename(columns={"symbol":"Symbol","title":"News Headline","Sentiment":"Sentiment 🧠"})
        news_df_display = news_df_display[["Symbol","News Headline","Sentiment 🧠"]]
        st.dataframe(news_df_display.style.map(color_sentiment, subset=["Sentiment 🧠"]), use_container_width=True, hide_index=True)
    else:
        st.info("No news available.")

# --------------------------
# CHATBOT
# --------------------------
with col_chat:
    st.subheader("🤖 Stock Chatbot")
    user_input = st.text_input("Ask about a stock or keyword:")

    if user_input:
        user_input_upper = user_input.upper()
        response = "Sorry, I couldn't find information."

        if user_input_upper in all_symbols:
            rec = pred_df[pred_df["symbol"] == user_input_upper]
            if not rec.empty:
                buy_pred = rec.iloc[0]["buy_pred"]
                sell_pred = rec.iloc[0]["sell_pred"]
                action = rec.iloc[0]["action"]
                response = f"**{user_input_upper}** → Model suggests **{action}** (Buy: {buy_pred*100:.1f}%, Sell: {sell_pred*100:.1f}%)"
            else:
                response = f"No predictions available for {user_input_upper}."
        else:
            news_match = news_df[news_df["title"].str.contains(user_input, case=False)]
            if not news_match.empty:
                headlines = news_match["title"].tolist()
                response = "Related news:\n" + "\n".join(headlines[:5])
        
        st.markdown(response)
