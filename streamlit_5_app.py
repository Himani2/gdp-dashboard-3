import os
import io
from datetime import datetime, timedelta
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
from fpdf import FPDF
from sqlalchemy import create_engine
import yfinance as yf

# -------------------------
# DATABASE CONFIGURATION
# -------------------------
db_user = "postgres"
db_password = "oX7IDNsZF1OrTOzS75Ek"
db_host = "database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com"
db_port = "5432"  # default PostgreSQL port
db_name = "capstone_project"


DB_URI = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
engine = create_engine(DB_URI)

# -------------------------
# LOAD DATA FUNCTION
# -------------------------
@st.cache_data(ttl=300)
def load_table(table_name):
    """Load table from database; return empty DataFrame if fails"""
    try:
        engine = create_engine(DB_URI)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        return df
    except Exception as e:
        st.warning(f"Failed to load {table_name}: {e}")
        return pd.DataFrame()

# -------------------------
# LOAD STOCKS AND PREDICTIONS
# -------------------------
stocks_df = load_table("stocks")
pred_df = load_table("buy_sell_predictions")
news_df = load_table("news_sentiment")
# -------------------------
# MARKET INFO
# -------------------------
IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()

weekend_warning = None
if today.weekday() >= 5:
    last_trading_day = today - timedelta(days=today.weekday()-4)
    weekend_warning = f"Market closed 🛑 Showing last Friday ({last_trading_day}) data"
else:
    last_trading_day = today

last_updated = now_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Watchlist & Stock Selection")
all_symbols = sorted(stocks_df["symbol"].unique()) if not stocks_df.empty else []
selected_symbols = st.sidebar.multiselect("Select Stocks for Chart", all_symbols, default=all_symbols[:3])
watchlist_symbols = st.sidebar.multiselect("Select Watchlist Stocks", all_symbols)

# -------------------------
# ALERTS SYSTEM
# -------------------------
ALERTS_CSV = "alerts_history.csv"
alerts = []

if not pred_df.empty:
    top_buy = pred_df[pred_df['action'].str.upper() == "BUY"].sort_values("buy_pred", ascending=False).head(3)
    top_sell = pred_df[pred_df['action'].str.upper() == "SELL"].sort_values("sell_pred", ascending=False).head(3)
    for _, row in top_buy.iterrows():
        alerts.append({"type": "BUY", "symbol": row["symbol"], "confidence": row["buy_pred"], "timestamp": row["timestamp"]})
    for _, row in top_sell.iterrows():
        alerts.append({"type": "SELL", "symbol": row["symbol"], "confidence": row["sell_pred"], "timestamp": row["timestamp"]})

# Save alerts to CSV
if alerts:
    alert_df = pd.DataFrame(alerts)
    if os.path.exists(ALERTS_CSV):
        alert_df.to_csv(ALERTS_CSV, mode='a', header=False, index=False)
    else:
        alert_df.to_csv(ALERTS_CSV, index=False)

# Display alert bell with count
st.markdown(f"### 🔔 Alerts ({len(alerts)})")
if alerts:
    alert_df_display = pd.DataFrame(alerts)
    alert_df_display["confidence"] = (alert_df_display["confidence"]*100).round(1).astype(str) + "%"
    st.dataframe(alert_df_display[["type","symbol","confidence","timestamp"]], use_container_width=True)

# -------------------------
# FETCH NIFTY AND INDICES FROM YAHOO FINANCE
# -------------------------
st.subheader("📊 Indian Market Indices")
indices_symbols = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY 500": "^CRSLDX"
}

indices_data = []
for name, symbol in indices_symbols.items():
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d")["Close"].iloc[-1]
    except:
        price = 0
    indices_data.append({"name": name, "symbol": symbol, "price": price})

index_df = pd.DataFrame(indices_data)
cols = st.columns(len(index_df))
for i, row in index_df.iterrows():
    cols[i].metric(label=row['name'], value=f"₹ {row['price']:.2f}")

    # ---------------------------
# Sidebar: Earnings Calendar
# ---------------------------
st.sidebar.title("📅 Earnings Calendar")

events_per_page = 5
if "calendar_page" not in st.session_state:
    st.session_state.calendar_page = 0

total_pages = (len(earnings_df) - 1) // events_per_page + 1

def prev_page():
    if st.session_state.calendar_page > 0:
        st.session_state.calendar_page -= 1

def next_page():
    if st.session_state.calendar_page < total_pages - 1:
        st.session_state.calendar_page += 1

col1, col2, col3 = st.sidebar.columns([1,6,1])
with col1:
    st.button("⬅️ Prev", on_click=prev_page)
with col3:
    st.button("Next ➡️", on_click=next_page)

start_idx = st.session_state.calendar_page * events_per_page
end_idx = start_idx + events_per_page
page_events = earnings_df.iloc[start_idx:end_idx]

st.sidebar.subheader(f"Events {start_idx+1}-{min(end_idx,len(earnings_df))} of {len(earnings_df)}")

for _, row in page_events.iterrows():
    event_time = row['date'].strftime("%b %d, %Y, %I:%M %p") if row['date'].hour != 0 else row['date'].strftime("%b %d, %Y")
    st.sidebar.markdown(
        f"**{row['symbol']}**  \n{event_time}"
    )


# -------------------------
# STOCK PRICE CHART
# -------------------------
st.subheader("📈 Stock Closing Prices")
if not stocks_df.empty and selected_symbols:
    df_chart = stocks_df[stocks_df['symbol'].isin(selected_symbols)]
    fig = px.line(df_chart, x="timestamp", y="close", color="symbol", title="Stock Closing Prices")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# WATCHLIST METRICS
# -------------------------
st.subheader("⭐ Watchlist Metrics")
if watchlist_symbols:
    watchlist_df = stocks_df[stocks_df['symbol'].isin(watchlist_symbols)].groupby('symbol').last().reset_index()
    cols = st.columns(len(watchlist_df))
    for i, row in watchlist_df.iterrows():
        cols[i].metric(label=row['symbol'], value=f"₹ {row['close']:.2f}", delta=f"{row['close']-row['open']:.2f}")


# -------------------------
# NEWS SENTIMENT CLEANING
# -------------------------
if not news_df.empty:
    # Map sentiment to standard labels
    def map_sentiment(sent):
        sent = str(sent).lower()
        if sent in ["positive","bullish","blusih"]:
            return "Bullish"
        elif sent in ["negative","bearish","berish"]:
            return "Bearish"
        else:
            return "Neutral"

    news_df["Sentiment"] = news_df["sentiment"].apply(map_sentiment)

    # Fill missing symbol if necessary
    if "symbol" not in news_df.columns:
        news_df["symbol"] = "UNKNOWN"

# -------------------------
# DISPLAY NEWS
# -------------------------
st.subheader("📰 Latest News & Sentiment")
if not news_df.empty:
    st.dataframe(news_df[['symbol','title','Sentiment']], use_container_width=True)
else:
    st.info("No news data available.")

# -------------------------
# BUY / SELL RECOMMENDATIONS
# -------------------------
st.subheader("💹 ML Trade Recommendations")
if not pred_df.empty:
    display_df = pred_df.copy()
    display_df["Buy %"] = (display_df["buy_pred"]*100).round(1)
    display_df["Sell %"] = (display_df["sell_pred"]*100).round(1)
    st.dataframe(display_df[["symbol","action","Buy %","Sell %","timestamp"]], use_container_width=True)

# -------------------------
# PDF REPORT FUNCTION
# -------------------------
def create_pdf_report(price_fig, pred_df, stocks_df, news_df, watchlist_symbols):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Indian Stock Monitor Daily Report",0,1,"C")
    pdf.set_font("Arial","",12)
    pdf.cell(0,10,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",0,1,"C")
    pdf.ln(5)

    # Watchlist Predictions
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"1. Watchlist Predictions",0,1,"L")
    pdf.set_font("Arial","",10)
    watchlist_pred = pred_df[pred_df['symbol'].isin(watchlist_symbols)]
    for _, row in watchlist_pred.iterrows():
        action = row['action']
        color=(0,128,0) if action.upper()=="BUY" else (255,0,0)
        pdf.set_text_color(*color)
        pdf.cell(0,6,f"{row['symbol']} | {row['action']} | ₹{row['price']:.2f} | {row.get('buy_pred',0)*100:.1f}%",0,1)
    pdf.set_text_color(0,0,0)
    pdf.ln(3)

    # Latest Prices
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"2. Watchlist Latest Prices",0,1,"L")
    pdf.set_font("Arial","",10)
    watchlist_latest = stocks_df[stocks_df['symbol'].isin(watchlist_symbols)].groupby('symbol').last().reset_index()
    for _, row in watchlist_latest.iterrows():
        pdf.cell(0,5,f"{row['symbol']} | Close ₹{row['close']:.2f} | Change {row['close']-row['open']:.2f}",0,1)
    pdf.ln(3)

    # News sentiment
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"3. News Sentiment",0,1,"L")
    pdf.set_font("Arial","",10)
    for _, row in news_df.iterrows():
        pdf.multi_cell(0,5,f"{row['symbol']} | {row['title']} | {row.get('sentiment','Neutral')}")

    # Chart
    try:
        img_bytes = price_fig.to_image(format="png", engine="kaleido")
        img_stream = io.BytesIO(img_bytes)
        pdf.image(img_stream, x=10, w=190)
    except:
        pdf.cell(0,10,"Chart could not render.",0,1)

    return pdf.output(dest='S').encode('latin-1', errors='ignore')

# -------------------------
# PDF DOWNLOAD BUTTON
# -------------------------
if st.button("📄 Download PDF Report"):
    if not stocks_df.empty and not pred_df.empty and selected_symbols:
        report_fig = px.line(stocks_df[stocks_df["symbol"].isin(selected_symbols)],
                             x="timestamp", y="close", color="symbol",
                             title="Stock Closing Prices")
        pdf_bytes = create_pdf_report(report_fig, pred_df, stocks_df, news_df, watchlist_symbols)
        st.download_button("Download PDF", data=pdf_bytes, file_name="Stock_Report.pdf", mime="application/pdf")
