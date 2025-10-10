# Save this code as 'app.py' and run with 'streamlit run app.py'

import streamlit as st
import yfinance as yf
import pandas as pd

# --- Back-end Data Fetching Class (Same as before) ---
class IndianStockAnalysis:
    def __init__(self, ticker_symbol):
        # Ensure the ticker has the correct Indian exchange suffix (e.g., .NS)
        if not ticker_symbol.endswith(('.NS', '.BO', '^NSEI', '^BSESN')):
            self.ticker = f"{ticker_symbol}.NS" if ticker_symbol else None
        else:
            self.ticker = ticker_symbol
            
        if self.ticker:
            try:
                self.stock = yf.Ticker(self.ticker)
                self.company_name = self.stock.info.get('longName', self.ticker)
                self.is_valid = True
            except Exception:
                self.is_valid = False
                st.error(f"Could not load data for ticker: {self.ticker}. Check the symbol.")
        else:
            self.is_valid = False

    def get_info(self):
        """Fetches basic company information."""
        if not self.is_valid: return ""
        info = self.stock.info
        
        # Format the output using Streamlit markdown/text
        output = f"## 🇮🇳 {self.company_name} Overview ({self.ticker})"
        output += "\n---\n"
        output += f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}\n"
        output += f"**Current Price:** ₹{info.get('currentPrice', 'N/A')}\n"
        output += f"**Market Cap:** ₹{info.get('marketCap', 'N/A'):,.0f}\n"
        output += f"**P/E Ratio:** {info.get('trailingPE', 'N/A'):.2f} | **Dividend Yield:** {info.get('dividendYield', 'N/A')}\n"
        
        st.markdown(output)

    def get_financial_statements(self, statement_type='Balance Sheet'):
        """Fetches a full annual financial statement."""
        if not self.is_valid: return
        st.subheader(f"Annual {statement_type}")
        
        try:
            if statement_type == 'Balance Sheet':
                df = self.stock.balance_sheet
            elif statement_type == 'Income Statement':
                df = self.stock.financials
            elif statement_type == 'Cash Flow':
                df = self.stock.cashflow
            
            if df is not None and not df.empty:
                st.dataframe(df.transpose()) # Show a clean, transposed DataFrame
            else:
                st.info(f"No annual {statement_type} data available.")
        except Exception as e:
            st.error(f"Error fetching annual statement: {e}")

    def get_quarterly_update(self, statement_type='Income Statement'):
        """Fetches the latest quarterly data."""
        if not self.is_valid: return
        st.subheader(f"Latest Quarterly {statement_type}")

        try:
            if statement_type == 'Income Statement':
                df = self.stock.quarterly_financials
            elif statement_type == 'Balance Sheet':
                df = self.stock.quarterly_balance_sheet
            
            if df is not None and not df.empty:
                # Show only the last 4 quarters for brevity
                st.dataframe(df.iloc[:, :4].transpose()) 
            else:
                st.info(f"No quarterly {statement_type} data available.")
        except Exception as e:
            st.error(f"Error fetching quarterly update: {e}")


# --- Streamlit App Front-End ---
st.set_page_config(layout="wide", page_title="Indian Stock Analyst 🇮🇳")
st.title("Indian Stock Analyst 🇮🇳")
st.markdown("Enter a stock symbol (e.g., RELIANCE, TCS, HDFCBANK) or an index (^NSEI, ^BSESN).")
st.markdown("---")


# --- Dynamic Prompt Suggestion Logic ---
def get_prompt_suggestions(user_input):
    """Generates suggested prompts based on partial user input."""
    suggestions = []
    
    # Standard Indian Tickers for Quick suggestions
    common_tickers = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK", "MARUTI", "WIPRO"]
    
    # Check for ticker-specific matches
    for ticker in common_tickers:
        if ticker.startswith(user_input.upper()):
            suggestions.append(f"show {ticker} info")
            suggestions.append(f"show {ticker} quarterly income")
    
    # Check for keyword matches
    if "quar" in user_input.lower():
        suggestions.append("show quarterly income statement")
        suggestions.append("show quarterly balance sheet")
    if "annu" in user_input.lower() or "statem" in user_input.lower():
        suggestions.append("show annual income statement")
        suggestions.append("show annual cash flow")
    if "bal" in user_input.lower():
        suggestions.append("show balance sheet")
    if "info" in user_input.lower():
        suggestions.append("show company info")
    if "comp" in user_input.lower():
        suggestions.append("compare TCS and INFY financials")
        
    return list(set(suggestions)) # Return unique suggestions


# --- User Input and Command Execution ---

# Input box for user query
user_query = st.text_input("Ask a question or enter a command (e.g., 'show RELIANCE annual cash flow')", key="query_input")

# Display suggested prompts based on the current input
if user_query:
    suggestions = get_prompt_suggestions(user_query)
    if suggestions:
        st.sidebar.markdown("**Suggested Commands** 👇")
        for suggestion in suggestions:
            # Use buttons to automatically populate the input field (better UX)
            if st.sidebar.button(suggestion.upper()):
                st.session_state.query_input = suggestion # Update the input box
                user_query = suggestion # Use the button text for execution

# Parse and Execute Command
if user_query:
    parts = user_query.lower().split()
    ticker_index = -1
    
    # 1. Identify the Ticker
    for i, part in enumerate(parts):
        # Simple check for a possible ticker (all caps or common Indian names)
        if part.upper() in ["RELIANCE", "TCS", "HDFCBANK", "INFY", "KOTAKBANK", "SBIN", "^NSEI", "^BSESN"] or (part.isupper() and len(part) >= 3):
            ticker_symbol = part
            ticker_index = i
            break
    else:
        # Default to a placeholder if no ticker is clearly identified, or let the user try again
        st.warning("Please specify a stock ticker (e.g., RELIANCE) in your command.")
        ticker_symbol = None

    if ticker_symbol:
        analyst = IndianStockAnalysis(ticker_symbol)

        if analyst.is_valid:
            # 2. Determine the Action
            action_found = False
            
            # Show Company Info
            if "info" in parts or "overview" in parts:
                analyst.get_info()
                action_found = True

            # Show Quarterly Data
            if "quarterly" in parts or "q" in parts:
                statement = "Income Statement"
                if "balance" in parts or "sheet" in parts:
                    statement = "Balance Sheet"
                elif "income" in parts or "financials" in parts:
                    statement = "Income Statement"
                
                analyst.get_quarterly_update(statement_type=statement)
                action_found = True
            
            # Show Annual Data (Financial Statements)
            elif "annual" in parts or "statement" in parts or "financials" in parts or "cash" in parts or "balance" in parts:
                statement = "Income Statement"
                if "balance" in parts or "sheet" in parts:
                    statement = "Balance Sheet"
                elif "cash" in parts or "flow" in parts:
                    statement = "Cash Flow"
                elif "income" in parts:
                    statement = "Income Statement"
                
                analyst.get_financial_statements(statement_type=statement)
                action_found = True

            # Fallback if a ticker was found but no clear action
            if not action_found:
                st.info(f"Command not recognized. Showing basic info for {analyst.company_name}.")
                analyst.get_info()
        else:
            st.error(f"Failed to fetch data for {ticker_symbol}. Check the symbol or try adding '.NS'.")