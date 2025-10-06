# -------------------------
# IMPORTS
# -------------------------
import os
import io
from datetime import datetime, timedelta
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
from fpdf import FPDF
import yfinance as yf
import feedparser

# -------------------------
# DATABASE / DATA SETUP
# -------------------------
# Example: Load your buy/sell predictions & news
# Replace with your actual database or CSV
pred_df = pd.read_csv("buy_sell_predictions.csv", encoding="utf-8")  # contains symbol, action, buy_pred, sell_pred, timestamp
news_df = pd.read_csv("news_sentiment.csv", encoding="utf-8")        # contains symbol, title, sentiment, timestamp

# -------------------------
# STREAMLIT DASHBOARD
# -------------------------
st.set_page_config(layout="wide", page_title="Indian Stock Monitor 📈")

IST = pytz.timezone("Asia/Kolkata")
now_ist = datetime.now(IST)
today = now_ist.date()

st.title("📈 Indian Stock Monitor")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.header("⚙️ Watchlist & Stock Selection")
all_symbols = sorted(pred_df["symbol"].unique())
selected_symbols = st.sidebar.multiselect("Select Stocks for Chart", all_symbols, default=all_symbols[:3])
watchlist_symbols = st.sidebar.multiselect("Select Watchlist Stocks", all_symbols)

# -------------------------
# FETCH NIFTY / SENSEX DATA
# -------------------------
@st.cache_data(ttl=300)
def fetch_index_data():
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTY BANK": "^NSEBANK",
    }
    df_list = []
    for name, ticker in indices.items():
        data = yf.download(ticker, period="30d", interval="1d")
        data.reset_index(inplace=True)
        data["Index"] = name
        df_list.append(data)
    return pd.concat(df_list, ignore_index=True)

index_df = fetch_index_data()

# -------------------------
# INDICES METRICS
# -------------------------
st.subheader("📊 Indian Market Indices")
latest_indices = index_df.groupby("Index").last().reset_index()
cols = st.columns(len(latest_indices))
for i, row in latest_indices.iterrows():
    cols[i].metric(label=row["Index"], value=f"₹ {row['Close']:.2f}", delta=f"{row['Close']-row['Open']:.2f}")

# -------------------------
# ALERTS & NOTIFICATIONS
# -------------------------
st.subheader("🚨 Alerts & Notifications")

alert_list = []

if not pred_df.empty:
    top_buy = pred_df[pred_df['action'].str.upper()=="BUY"].sort_values("buy_pred",ascending=False).head(1)
    top_sell = pred_df[pred_df['action'].str.upper()=="SELL"].sort_values("sell_pred",ascending=False).head(1)
    
    if not top_buy.empty:
        alert = f"Top Buy Alert: {top_buy.iloc[0]['symbol']} ({top_buy.iloc[0]['buy_pred']*100:.1f}%)"
        alert_list.append(alert)
        st.info(alert)
    if not top_sell.empty:
        alert = f"Top Sell Alert: {top_sell.iloc[0]['symbol']} ({top_sell.iloc[0]['sell_pred']*100:.1f}%)"
        alert_list.append(alert)
        st.warning(alert)

# Save alerts to CSV
if alert_list:
    alert_df = pd.DataFrame({
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * len(alert_list),
        "alert": alert_list
    })
    if os.path.exists("alerts.csv"):
        alert_df.to_csv("alerts.csv", mode="a", header=False, index=False, encoding="utf-8")
    else:
        alert_df.to_csv("alerts.csv", index=False, encoding="utf-8")

# -------------------------
# WATCHLIST METRICS
# -------------------------
st.subheader("⭐ Watchlist Metrics")
if watchlist_symbols:
    watchlist_data = yf.download([f"{s}.NS" for s in watchlist_symbols], period="30d", interval="1d", group_by='ticker')
    cols = st.columns(len(watchlist_symbols))
    for i, sym in enumerate(watchlist_symbols):
        if sym+".NS" in watchlist_data:
            df = watchlist_data[sym+".NS"]
            latest = df.iloc[-1]
            delta = latest['Close'] - latest['Open']
            cols[i].metric(label=sym, value=f"₹ {latest['Close']:.2f}", delta=f"{delta:.2f}")

# -------------------------
# STOCK CHARTS
# -------------------------
st.subheader("📊 Stock Price Trend")
if selected_symbols:
    chart_data = yf.download([f"{s}.NS" for s in selected_symbols], period="30d", interval="1d", group_by='ticker')
    fig = px.line()
    for s in selected_symbols:
        df = chart_data[s+".NS"].reset_index()
        fig.add_scatter(x=df['Date'], y=df['Close'], mode='lines', name=s)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# NEWS SENTIMENT
# -------------------------
st.subheader("📰 Latest News & Sentiment")
if not news_df.empty:
    def map_sentiment(sent):
        sent = str(sent).lower()
        if sent in ["positive","bullish","blusih"]: return "Bullish"
        elif sent in ["negative","bearish","berish"]: return "Bearish"
        else: return "Neutral"
    news_df["Sentiment"] = news_df["sentiment"].apply(map_sentiment)
    st.dataframe(news_df[['symbol','title','Sentiment','timestamp']], use_container_width=True)

# -------------------------
# BUY/SELL RECOMMENDATIONS
# -------------------------
st.subheader("💹 Buy/Sell Recommendations")
display_df = pred_df.copy()
display_df['Action'] = display_df['action'].apply(lambda x: "⬆️ Buy" if x.upper()=="BUY" else ("⬇️ Sell" if x.upper()=="SELL" else "⏸️ No Trade"))
st.dataframe(display_df[['symbol','Action','buy_pred','sell_pred','timestamp']], use_container_width=True)

# -------------------------
# PDF REPORT DOWNLOAD
# -------------------------
def create_pdf_report(price_fig, pred_df, news_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Indian Stock Monitor Report",0,1,"C")
    pdf.set_font("Arial","",12)
    pdf.cell(0,10,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",0,1,"C")
    pdf.ln(5)
    
    # Buy/Sell Table
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"Buy/Sell Recommendations",0,1)
    pdf.set_font("Arial","",10)
    for _, row in pred_df.iterrows():
        pdf.cell(0,5,f"{row['symbol']} | {row['action']} | Buy%: {row['buy_pred']*100:.1f}% | Sell%: {row['sell_pred']*100:.1f}% | {row['timestamp']}",0,1)
    pdf.ln(5)
    
    # News Table
    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,"News Sentiment",0,1)
    pdf.set_font("Arial","",10)
    for _, row in news_df.iterrows():
        pdf.multi_cell(0,5,f"{row['symbol']} | {row['title']} | {row['sentiment']} | {row['timestamp']}")
    
    return pdf.output(dest='S').encode('utf-8')

if st.button("📄 Download PDF Report"):
    pdf_bytes = create_pdf_report(fig, pred_df, news_df)
    st.download_button("Download PDF", data=pdf_bytes, file_name="Stock_Report.pdf", mime="application/pdf")
