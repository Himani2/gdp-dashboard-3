import streamlit as st
import os
import pandas as pd
import yfinance as yf
from google import genai
from google.genai.types import FunctionDeclaration, Type, Tool
from typing import List, Dict, Any, Optional

# --- 1. Agent Prompts and Database Setup ---
DATA_EXTRACTION_PROMPT = """You are a Data Extraction Agent specializing in financial and sentiment data.

Your capabilities:
1. Database Operations:
   - list_postgres_tables(): Explore available tables
   - describe_postgres_table(table_name): Get table structure and samples
   - answer_postgres_question(question): Smart NL query processing
   - execute_postgres_query(query): Run direct SQL queries
   - load_postgres_table(table_name, limit): Load full tables

2. Stock Data:
   - get_india_daily_and_rsi(symbol, period, interval): Fetch Indian stock data
   - analyze_dataframe(df_id, analysis_type): Analyze loaded data

Your role:
- Understand user data requirements
- Extract relevant data from databases and stock APIs
- Perform initial analysis and data quality checks
- Prepare clean datasets for the plotting agent
- Always use log_step to document your actions

Guidelines:
- Start by exploring available data sources
- Combine multiple data sources when relevant
- Provide summary statistics and data insights
- Store all extracted data with clear df_ids
- Communicate findings clearly to the plotting agent
"""

PLOTTING_PROMPT = """You are a Plotting Agent specializing in financial data visualizations.

Your capabilities:
1. Standard Plots:
   - plot_df_matplotlib(df_id, plot_type, params): Create various chart types
   - Supported types: "line", "ema_overlay", "candlestick", "sentiment_timeline", "correlation_heatmap"

2. Advanced Visualizations:
   - create_dashboard_plot(df_ids, plot_configs): Multi-panel dashboards

Your role:
- Create compelling visualizations from extracted data
- Choose appropriate chart types for different data
- Enhance plots with proper styling and annotations
- Create dashboards combining multiple datasets
- Always use log_step to document your actions

Plot Type Guidelines:
- line: Basic time series or single variable plots
- ema_overlay: Stock prices with EMA indicators
- candlestick: OHLC stock data visualization
- sentiment_timeline: Sentiment scores over time
- correlation_heatmap: Correlation matrix visualization
- dashboard: Multiple plots in one view

Focus on clarity, professional appearance, and actionable insights.
"""

# --- 2. Tool/Function Implementations ---

# In a real app, you would use 'psycopg2' or 'sqlalchemy' here.
# For this example, we mock the database and tool outputs.

# Mock Database Tables (based on user's prompt)
MOCK_TABLES = {
    "news": ["source", "title", "description", "publish_datetime"],
    "stocks": ["timestamp", "open", "high", "low", "close", "volume", "symbol"],
    "buy_Sell_prediction": ["timestamp", "symbol", "buy_pred", "sell_pred", "action", "price", "target_price", "stop_loss"],
}

# In-memory store for dataframes created by the Data Agent
DF_STORE = {}

def list_postgres_tables() -> str:
    """Explores available database tables."""
    st.session_state.log_step("Data Extraction Agent: Listing database tables.")
    return f"Available tables: {', '.join(MOCK_TABLES.keys())}"

def get_india_daily_and_rsi(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches daily stock data and calculates 14-day RSI for an Indian stock.
    Indian symbols require the .NS suffix (e.g., RELIANCE.NS).
    """
    st.session_state.log_step(f"Data Extraction Agent: Fetching {symbol} data with period={period} and interval={interval}.")
    
    # 1. Format symbol for Yahoo Finance (Indian stocks)
    if not symbol.upper().endswith(".NS"):
        symbol += ".NS"

    # 2. Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return f"Error: Could not find data for symbol {symbol}."

    # 3. Calculate RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Store the dataframe
    df_id = f"stock_data_{symbol.replace('.NS', '')}"
    DF_STORE[df_id] = df.reset_index().rename(columns={'Date': 'timestamp'})
    
    sample_data = DF_STORE[df_id].tail(3).to_markdown(index=False)
    
    return f"Successfully loaded daily and RSI data for {symbol}.\nDataFrame ID: `{df_id}`. Sample Data:\n{sample_data}"

def plot_df_matplotlib(df_id: str, plot_type: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Simulates plotting a dataframe using the Plotting Agent.
    In a real app, this would generate and return a plot image.
    """
    st.session_state.log_step(f"Plotting Agent: Creating a {plot_type} plot for dataframe ID `{df_id}`.")
    
    if df_id not in DF_STORE:
        return f"Error: Dataframe with ID `{df_id}` not found."

    df = DF_STORE[df_id]

    # Mock the plot generation by displaying the chart in Streamlit
    if plot_type == "line":
        st.line_chart(df.set_index('timestamp')[['Close', 'RSI'] if 'RSI' in df.columns else 'Close'])
    elif plot_type == "candlestick":
        # Simplified representation for the chatbot UI
        st.write("Visualizing **Candlestick** chart...")
        st.dataframe(df.tail(5)) 
    elif plot_type == "ema_overlay":
        st.write("Visualizing **EMA Overlay** (simulated: showing Close vs RSI)")
        st.line_chart(df.set_index('timestamp')[['Close', 'RSI']])
    else:
        st.warning(f"Plot type `{plot_type}` is simulated. Showing the raw data table.")
        st.dataframe(df)

    return f"Successfully generated a **{plot_type}** visualization for dataframe `{df_id}`."

# --- 3. Gemini Agent Functions and Tool Definitions ---

def get_agent_tools(agent_type: str) -> List[Tool]:
    """Dynamically creates the tool list for the Gemini model."""
    if agent_type == "data":
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="list_postgres_tables",
                        description="Explore available tables in the PostgreSQL database.",
                        parameters=Type.object_type(),
                    ),
                    FunctionDeclaration(
                        name="get_india_daily_and_rsi",
                        description="Fetch Indian stock's daily OHLCV data and calculate the 14-day Relative Strength Index (RSI). Requires the stock symbol.",
                        parameters=Type.object_type(
                            properties={
                                "symbol": Type.string_type(description="The stock ticker symbol (e.g., 'RELIANCE', 'TCS')."),
                                "period": Type.string_type(description="Period for data: '1mo', '3mo', '1y'. Default is '1mo'."),
                                "interval": Type.string_type(description="Data interval: '1d', '1wk'. Default is '1d'."),
                            },
                            required=["symbol"]
                        )
                    )
                ]
            )
        ]
    elif agent_type == "plot":
        return [
            Tool(
                function_declarations=[
                    FunctionDeclaration(
                        name="plot_df_matplotlib",
                        description="Creates a visualization from a loaded DataFrame ID. Supported types: line, ema_overlay, candlestick, sentiment_timeline.",
                        parameters=Type.object_type(
                            properties={
                                "df_id": Type.string_type(description="The unique ID of the DataFrame to plot, provided by the Data Agent (e.g., 'stock_data_RELIANCE')."),
                                "plot_type": Type.string_type(description="The type of plot to create (e.g., 'candlestick', 'line', 'ema_overlay')."),
                                "params": Type.object_type(description="Optional dict of plotting parameters (e.g., {'color': 'blue'}).")
                            },
                            required=["df_id", "plot_type"]
                        )
                    )
                ]
            )
        ]
    return []

def run_agent_workflow(client: genai.Client, user_query: str):
    """Manages the multi-step, multi-agent conversation (Data -> Plot)."""
    
    # 1. Start with Data Extraction Agent
    agent = "Data Extraction Agent"
    prompt = f"{DATA_EXTRACTION_PROMPT}\n\nUser Request: {user_query}"
    model = "gemini-2.5-flash"
    
    # Display initial thought
    st.session_state.log_step(f"Starting workflow with **{agent}**...")
    
    for _ in range(3): # Max 3 tool-use iterations to prevent loops
        
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                tools=get_agent_tools("data" if agent == "Data Extraction Agent" else "plot"),
                temperature=0.0,
            ),
        )
        
        # Check for tool call
        if response.function_calls:
            function_call = response.function_calls[0]
            function_name = function_call.name
            args = dict(function_call.args)
            
            # Execute the function
            if function_name == "list_postgres_tables":
                tool_output = list_postgres_tables()
            elif function_name == "get_india_daily_and_rsi":
                tool_output = get_india_daily_and_rsi(**args)
            elif function_name == "plot_df_matplotlib":
                tool_output = plot_df_matplotlib(**args)
            else:
                tool_output = f"Error: Unknown function {function_name}"
            
            # Send tool output back to the model
            tool_output_part = genai.types.Part.from_function_result(
                name=function_name,
                response={"result": tool_output},
            )
            
            prompt = f"{response.text}\n{tool_output_part.text}"
            st.session_state.log_step(f"**{agent}** called `{function_name}({args})` -> Output received.")
            
        else:
            # Data Extraction Agent finishes, pass to Plotting Agent
            if agent == "Data Extraction Agent":
                # Check if data was extracted before passing to Plotting Agent
                if not DF_STORE:
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    return # Exit if no data was fetched
                
                # Transition to Plotting Agent
                agent = "Plotting Agent"
                data_summary = f"Available DataFrames: {', '.join(DF_STORE.keys())}"
                
                # New prompt for the Plotting Agent
                prompt = (
                    f"{PLOTTING_PROMPT}\n\n"
                    f"User Request: {user_query}\n"
                    f"Data Extraction Agent Summary: {response.text}\n"
                    f"DATA: {data_summary}\n"
                    "Now, generate a relevant plot using the available data."
                )
                st.session_state.log_step(f"Transitioning to **{agent}**...")
            
            # Plotting Agent finishes, present final answer
            elif agent == "Plotting Agent":
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                return
                
    # If loop finishes without returning (e.g., max iterations reached)
    st.session_state.messages.append({"role": "assistant", "content": "I've reached my maximum processing steps. Please try a simpler query or check the logs for the last action."})


# --- 4. Streamlit UI Setup ---

# Initialize the client (handles GEMINI_API_KEY environment variable)
try:
    client = genai.Client()
except Exception as e:
    st.error(f"Failed to initialize Gemini Client. Check GEMINI_API_KEY environment variable. Error: {e}")
    st.stop()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log" not in st.session_state:
    st.session_state.log = []

def log_step(message: str):
    """Function to add a step to the logging section."""
    st.session_state.log.append(message)

# Inject the logging function into session state for agents to use
st.session_state.log_step = log_step


# --- Streamlit Layout ---
st.title("📈 Indian Stock Analysis Chatbot")
st.subheader("Powered by Gemini Agents & Yahoo Finance")

# Main Chat Area
chat_placeholder = st.container()

with chat_placeholder:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Analyze Reliance stock price and generate a candlestick chart."):
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run the agent workflow (this updates the messages in session state)
        with st.spinner("Agents are working..."):
            run_agent_workflow(client, prompt)
            
        # Rerun to update the display with the new messages
        st.experimental_rerun()


# --- Sidebar and Debugging ---
with st.sidebar:
    st.header("Debug and Logs")
    if st.button("Clear History and Data"):
        st.session_state.messages = []
        st.session_state.log = []
        DF_STORE.clear()
        st.experimental_rerun()

    st.subheader("Agent Log")
    log_container = st.container()
    for entry in st.session_state.log:
        log_container.markdown(f"- {entry}")

    st.subheader("DataFrame Store")
    if DF_STORE:
        for df_id, df in DF_STORE.items():
            st.write(f"**{df_id}** ({len(df)} rows)")
    else:
        st.write("No DataFrames loaded.")