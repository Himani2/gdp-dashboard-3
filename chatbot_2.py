import streamlit as st
import os
import uuid
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Any, Optional, List

# --- CONFIGURE LOGGING AND PATHS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AGENT AND MODEL CONFIGURATION ---
# 🚨 REQUIRED: UPDATE THIS PATH TO YOUR DOWNLOADED GGUF MODEL FILE
LOCAL_MODEL_PATH = "path/to/your/tiny-llama-3-8b.gguf" 

MODEL_CONFIG = {
    'max_new_tokens': 1024,
    'temperature': 0.1,
    'context_length': 4096,
}

# --- MOCK & LANGCHAIN CORE COMPONENTS ---

# Mocking external/LangGraph components for the workflow structure
class DualAgentState(dict): pass
START = "start"
END = "end"

class StateGraph:
    """Mock StateGraph class for workflow definition."""
    def __init__(self, state_class): pass
    def add_node(self, name, func): pass
    def add_edge(self, source, dest): pass
    def add_conditional_edges(self, source, condition, mapping): pass
    
    def compile(self): return self
    
    def invoke(self, state): 
        """Simplified sequential invocation for Streamlit/local demo."""
        # 1. Data Extraction Phase
        data_llm, data_tools = make_data_extraction_agent()
        data_result = data_llm(state)
        # 2. Plotting Phase
        plot_llm, plot_tools = make_plotting_agent()
        # Merge data results for plotting agent
        plot_state = {**data_result, "messages": data_result["messages"]}
        plot_result = plot_llm(plot_state)
        # 3. Final Summary
        final_state = {**plot_state, **plot_result}
        return create_final_summary(final_state)

class ToolNode:
    """Mock ToolNode to represent tool execution in the graph."""
    def __init__(self, tools):
        self.tools = {t.__name__: t for t in tools}

    def __call__(self, state: DualAgentState):
        """Mock tool execution: identifies the tool call from text and executes."""
        last_content = state["messages"][-1].content.strip()
        
        # Simple parsing logic (must match the prompt instruction)
        match = re.match(r'(\w+)\s*\((.*)\)', last_content)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Simple argument parsing (VERY rudimentary for demo)
            args = {}
            for pair in args_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Attempt to clean and evaluate the value
                    value = value.strip().strip('"').strip("'")
                    args[key.strip()] = value
            
            if tool_name in self.tools:
                tool_func = self.tools[tool_name]
                # Log the attempted tool call
                logger.info(f"Executing tool: {tool_name} with args: {args}")
                # Execute the tool and get result
                tool_result = tool_func(**args)
                
                # Append tool result to messages for the next LLM
                tool_message = {"role": "tool", "content": str(tool_result)}
                return {"messages": state["messages"] + [tool_message]}

        # If parsing fails or tool not found, return original state
        return state

class SystemMessage:
    def __init__(self, content): self.content = content

# Mock Model Response structure (for compatibility with original flow)
class MockToolResponse:
    def __init__(self, content):
        self.content = content
        self.tool_calls = [] # Always empty for local model

# Hugging Face LLM Wrapper
from langchain_community.llms import CTransformers
def create_local_llm(model_path: str, config: dict):
    """Initializes CTransformers for GGUF model."""
    if not os.path.exists(model_path):
        st.error(f"🚨 GGUF Model file not found at: `{model_path}`. Please update the `LOCAL_MODEL_PATH`.")
        return None

    try:
        # NOTE: Model type must match your downloaded model (e.g., 'llama', 'mistral')
        llm = CTransformers(
            model=model_path,
            model_type="llama", 
            config=config,
            verbose=False
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load CTransformers model. Check model type and dependencies. Error: {e}")
        return None

# --- MOCK DATA REGISTRY & TOOL FUNCTIONS ---
DF_REGISTRY = {}
def get_df(df_id: str) -> pd.DataFrame:
    """Mock function to retrieve DataFrame from registry or create mock data."""
    if df_id not in DF_REGISTRY:
        # Create a mock DataFrame
        dates = pd.date_range(end=pd.Timestamp.now(), periods=150)
        data = np.random.randn(150, 5).cumsum(axis=0) + 100
        df = pd.DataFrame(data, index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df['Sentiment'] = np.random.uniform(-1, 1, 150)
        DF_REGISTRY[df_id] = df
        return df
    return DF_REGISTRY[df_id]

def log_step(*args, **kwargs): logger.info(f"LOG: {args}, {kwargs}")
def list_postgres_tables(): return ['sentiment_data', 'user_feedback']
def describe_postgres_table(name): return f"Mock description for {name}: Columns include Date, Score."
def load_postgres_table(name, limit): 
    df_id = f"db_data:{name}"
    get_df(df_id) # Creates mock data
    return {"df_id": df_id, "shape": get_df(df_id).shape}
def execute_postgres_query(query): 
    df_id = f"query_result:{uuid.uuid4().hex[:4]}"
    get_df(df_id)
    return {"df_id": df_id}
def answer_postgres_question(question): return "The mock database indicates a positive trend."
def get_india_daily_and_rsi(symbol, period="1y", interval="1d"): 
    df_id = f"stock_data:{symbol}"
    get_df(df_id)
    return {"df_id": df_id}
def analyze_dataframe(df_id, analysis_type): return f"Mock analysis of {df_id}: High correlation found."

# =============================================================================
# PLOTTING TOOLS (Functions remain as defined by the user)
# =============================================================================
# ... (plot_df_matplotlib and create_dashboard_plot code goes here) ...

def plot_df_matplotlib(
    df_id: str,
    plot_type: str = "line",
    params: Optional[Dict[str, Any]] = None,
    width: Optional[int] = 12,
    height: Optional[int] = 8,
    work_dir: Optional[str] = None,
    return_thumbnail: bool = True
) -> Dict[str, Any]:
    """Create matplotlib plots from DataFrame."""
    # ... (Your plot_df_matplotlib implementation) ...
    try:
        df = get_df(df_id)
        params = params or {}
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, ax = plt.subplots(figsize=(width, height))
        except:
             return {"error": "Matplotlib or Seaborn not available for plotting.", "success": False}

        
        if plot_type == "line":
            y_col = params.get("y", "Close")
            window = params.get("window", min(len(df), 120))
            
            if y_col not in df.columns:
                available_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
                y_col = available_cols[0] if available_cols else 'Close'
            
            data = df[y_col].tail(window)
            ax.plot(data.index, data.values, linewidth=2, label=y_col, color='steelblue')
            ax.set_title(f"{y_col} - Last {window} periods", fontsize=14, fontweight='bold')
            
        elif plot_type == "ema_overlay":
            spans = params.get("spans", [12, 26, 50])
            window = params.get("window", min(len(df), 120))
            
            close_data = df['Close'].tail(window)
            ax.plot(close_data.index, close_data.values, label='Close', linewidth=2.5, color='navy')
            
            colors = ['crimson', 'forestgreen', 'darkorange', 'purple', 'brown']
            for i, span in enumerate(spans):
                if len(close_data) >= span:
                    ema = close_data.ewm(span=span).mean()
                    color = colors[i % len(colors)]
                    ax.plot(ema.index, ema.values, label=f'EMA{span}', 
                            linestyle='--', color=color, linewidth=2, alpha=0.8)
            
            ax.set_title(f"Price with EMA Overlays - Last {window} periods", fontsize=14, fontweight='bold')
            
        elif plot_type == "candlestick":
            window = params.get("window", min(len(df), 60))
            data = df.tail(window)
            
            up_days = data['Close'] >= data['Open']
            down_days = ~up_days
            
            ax.bar(data.index[up_days], 
                    (data['Close'] - data['Open'])[up_days],
                    bottom=data['Open'][up_days],
                    color='lightgreen', edgecolor='darkgreen', alpha=0.8, width=0.8)
            ax.bar(data.index[down_days], 
                    (data['Close'] - data['Open'])[down_days],
                    bottom=data['Open'][down_days], 
                    color='lightcoral', edgecolor='darkred', alpha=0.8, width=0.8)
            
            ax.set_title(f"Candlestick Chart - Last {window} periods", fontsize=14, fontweight='bold')
            
        elif plot_type == "sentiment_timeline":
            date_cols = [c for c in df.columns if any(word in c.lower() for word in ['date', 'time', 'timestamp'])]
            sentiment_cols = [c for c in df.columns if any(word in c.lower() for word in ['sentiment', 'score', 'polarity'])]
            
            if date_cols and sentiment_cols:
                date_col = date_cols[0]
                sentiment_col = sentiment_cols[0]
                
                df_plot = df.copy()
                df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')
                df_plot = df_plot.dropna(subset=[date_col]).sort_values(date_col)
                
                window = params.get("window", min(len(df_plot), 100))
                plot_data = df_plot.tail(window)
                
                ax.plot(plot_data[date_col], plot_data[sentiment_col], 
                        linewidth=2, marker='o', markersize=4, alpha=0.7, color='steelblue')
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_title(f"Sentiment Timeline - Last {window} entries", fontsize=14, fontweight='bold')
                ax.set_ylabel("Sentiment Score")
            else:
                return {"error": "No date or sentiment columns found for timeline plot", "success": False}
                
        elif plot_type == "correlation_heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                            square=True, ax=ax, fmt='.2f')
                ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
            else:
                return {"error": "Need at least 2 numeric columns for correlation", "success": False}
                
        else:
            close_data = df['Close'].tail(min(len(df), 120)) if 'Close' in df.columns else df.iloc[:, 0].tail(120)
            ax.plot(close_data.index, close_data.values, linewidth=2, color='steelblue')
            ax.set_title("Data Plot", fontsize=14, fontweight='bold')
        
        # Enhanced formatting
        ax.grid(True, alpha=0.3, linestyle=':')
        if ax.get_legend_handles_labels()[0]: 
            ax.legend(frameon=True, fancybox=True, shadow=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), "temp_plots")
        os.makedirs(work_dir, exist_ok=True)
        
        filename = f"plot-{uuid.uuid4().hex}.png"
        filepath = os.path.join(work_dir, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        return {
            "image_path": f"file://{filepath}",
            "thumbnail": "present" if return_thumbnail else "absent",
            "meta": {"df_id": df_id, "plot_type": plot_type, "shape": df.shape, "columns": df.columns.tolist()}
        }
        
    except Exception as e:
        error_msg = f"Plot creation error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def create_dashboard_plot(
    df_ids: List[str],
    plot_configs: List[Dict[str, Any]],
    width: Optional[int] = 15,
    height: Optional[int] = 10,
    work_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Create a multi-panel dashboard with different plots."""
    # ... (Your create_dashboard_plot implementation) ...
    try:
        n_plots = len(df_ids)
        if n_plots != len(plot_configs):
            return {"error": "Number of df_ids must match plot_configs", "success": False}
        
        if n_plots <= 2:
            rows, cols = 1, n_plots
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(rows, cols, figsize=(width, height))
        except:
             return {"error": "Matplotlib or Seaborn not available for plotting.", "success": False}
             
        if n_plots == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (df_id, config) in enumerate(zip(df_ids, plot_configs)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            df = get_df(df_id)
            plot_type = config.get('plot_type', 'line')
            
            if plot_type == 'line':
                y_col = config.get('y_col', 'Close')
                if y_col in df.columns:
                    data = df[y_col].tail(config.get('window', 60))
                    ax.plot(data.index, data.values, linewidth=2)
                    ax.set_title(f"{y_col} ({df_id.split(':')[1] if ':' in df_id else df_id})")
            
            elif plot_type == 'sentiment':
                sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower() or 'score' in c.lower()]
                if sentiment_cols:
                    data = df[sentiment_cols[0]].tail(config.get('window', 50))
                    ax.plot(data.index, data.values, linewidth=2, marker='o', markersize=3)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax.set_title(f"Sentiment ({df_id.split('_')[1] if '_' in df_id else 'Data'})")
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle("Financial Analysis Dashboard", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), "temp_plots")
        os.makedirs(work_dir, exist_ok=True)
        
        filename = f"dashboard-{uuid.uuid4().hex}.png"
        filepath = os.path.join(work_dir, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return {
            "image_path": f"file://{filepath}",
            "plot_count": n_plots,
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Dashboard creation failed: {str(e)}", "success": False}

# =============================================================================
# AGENT PROMPTS (Unchanged)
# =============================================================================

DATA_EXTRACTION_PROMPT = """You are a Data Extraction Agent specializing in financial and sentiment data.

Your capabilities:
... (list of tools) ...

Your role:
- Understand user data requirements
- Extract relevant data from databases and stock APIs
- Prepare clean datasets for the plotting agent
- Always use log_step to document your actions

Guidelines:
- Start by exploring available data sources
- Combine multiple data sources when relevant
- Store all extracted data with clear df_ids
- Communicate findings clearly to the plotting agent
"""

PLOTTING_PROMPT = """You are a Plotting Agent specializing in financial data visualizations.

Your capabilities:
... (list of tools) ...

Your role:
- Create compelling visualizations from extracted data
- Choose appropriate chart types for different data
- Enhance plots with proper styling and annotations
- Create dashboards combining multiple datasets
- Always use log_step to document your actions

Focus on clarity, professional appearance, and actionable insights.
"""

# =============================================================================
# AGENT FACTORIES (Updated for CTransformers/Prompt Engineering)
# =============================================================================

@st.cache_resource
def make_data_extraction_agent(model_name: str = "llama3-8b-local"):
    llm = create_local_llm(LOCAL_MODEL_PATH, MODEL_CONFIG)
    if not llm: return None, ToolNode([]) 

    tools = [
        log_step, list_postgres_tables, describe_postgres_table, load_postgres_table,
        execute_postgres_query, answer_postgres_question, get_india_daily_and_rsi, analyze_dataframe
    ]
    toolnode = ToolNode(tools)
    tool_list = "\n".join([f"- {t.__name__}: {t.__doc__.split('.')[0]}" for t in tools])
    
    def data_extraction_llm(state: DualAgentState):
        user_input = state["messages"][-1]["content"] 
        
        full_prompt_text = f"""{DATA_EXTRACTION_PROMPT}

AVAILABLE TOOLS:
{tool_list}

INSTRUCTION: 
Based on the user request, determine the best tool to use. Respond ONLY with the tool call 
in the format: '<tool_name>(<param1>="<value>", <param2>="<value>")'. 
If no tool is needed (e.g., if you already have the data or analysis is requested), respond with a thoughtful analysis.

USER REQUEST: {user_input}
"""
        resp_content = llm(full_prompt_text)
        
        # Mock extracted data IDs based on tool response (simplified for demo)
        extracted_ids = state.get("extracted_data_ids", [])
        if "get_india_daily_and_rsi" in resp_content: extracted_ids.append("stock_data:TCS")
        
        return {
            "messages": [MockToolResponse(resp_content)],
            "data_extraction_thought": resp_content,
            "extracted_data_ids": list(set(extracted_ids))
        }
    
    return data_extraction_llm, toolnode

@st.cache_resource
def make_plotting_agent(model_name: str = "llama3-8b-local"):
    llm = create_local_llm(LOCAL_MODEL_PATH, MODEL_CONFIG)
    if not llm: return None, ToolNode([])

    tools = [log_step, plot_df_matplotlib, create_dashboard_plot]
    toolnode = ToolNode(tools)
    tool_list = "\n".join([f"- {t.__name__}: {t.__doc__.split('.')[0]}" for t in tools])

    def plotting_llm(state: DualAgentState):
        user_input = state["messages"][-1]["content"]
        data_ids = state.get("extracted_data_ids", [])
        data_info = f"Available DataFrames: {', '.join(data_ids)}" if data_ids else "No DataFrames available."

        full_prompt_text = f"""{PLOTTING_PROMPT}

AVAILABLE PLOTTING TOOLS:
{tool_list}

{data_info}

INSTRUCTION: 
Based on the user request and available data, call the plotting tool. 
Respond ONLY with the tool call in the format: '<tool_name>(<param1>="<value>", <param2>="<value>")'. 
If plotting is complete or impossible, respond with a final conclusion/error message.

USER REQUEST: {user_input}
"""
        resp_content = llm(full_prompt_text)
        
        # Mock plot paths for demo (simplified)
        plot_paths = state.get("plot_paths", [])
        if "plot_df_matplotlib" in resp_content or "create_dashboard_plot" in resp_content:
             plot_paths.append(f"file:///content/temp_plots/plot-{uuid.uuid4().hex[:8]}.png")
        
        return {
            "messages": [MockToolResponse(resp_content)],
            "plotting_thought": resp_content,
            "plot_paths": plot_paths
        }
    
    return plotting_llm, toolnode

# =============================================================================
# ROUTING AND SUMMARY (Modified for text parsing)
# =============================================================================

def route_tools_or_next(state: DualAgentState):
    """Route to tools if tool calls present (via text parsing), otherwise continue."""
    last_message = state["messages"][-1]
    last_content = getattr(last_message, 'content', '').strip()
    
    # Check for the tool call pattern: function_name(...)
    tool_call_pattern = re.compile(r'^\s*(\w+)\s*\((.*)\)\s*$', re.DOTALL)
    
    if tool_call_pattern.match(last_content):
        return "tools"
    
    return END

def create_final_summary(state: DualAgentState):
    """Create final summary combining both agents' work."""
    data_thought = state.get("data_extraction_thought", "No data extraction thought generated.")
    plot_thought = state.get("plotting_thought", "No plotting thought generated.")
    extracted_ids = state.get("extracted_data_ids", [])
    plot_paths = state.get("plot_paths", [])
    
    summary = f"""
### DUAL-AGENT ANALYSIS SUMMARY

| Phase | Insights |
| :--- | :--- |
| **Data Extraction** | {data_thought[:60] + '...' if len(data_thought) > 60 else data_thought} |
| **Plotting** | {plot_thought[:60] + '...' if len(plot_thought) > 60 else plot_thought} |

**Extracted DataFrames:** {len(extracted_ids)}
{chr(10).join([f"  - {df_id}" for df_id in extracted_ids[:3]]) if extracted_ids else "  - None"}

**Generated Visualizations:** {len(plot_paths)}
{chr(10).join([f"  - {path}" for path in plot_paths[:2]]) if plot_paths else "  - None"}

**STATUS:** Analysis Complete.
(Note: Actual plot image is generated in the local `/temp_plots` directory.)
""".strip()
    
    return {"final_summary": summary, **state}

# =============================================================================
# GRAPH BUILDER
# =============================================================================

@st.cache_resource
def build_dual_agent_graph(model_name: str = "llama3-8b-local"):
    """Build the dual-agent graph (simplified for local execution)."""
    # NOTE: Error handling for missing LLM
    try:
        data_llm, data_tools = make_data_extraction_agent(model_name)
        plot_llm, plot_tools = make_plotting_agent(model_name)
    except Exception as e:
        st.error("Failed to initialize agents. See console for model loading errors.")
        return None

    graph = StateGraph(DualAgentState)
    
    graph.add_node("data_extraction_llm", data_llm)
    graph.add_node("plotting_llm", plot_llm)
    graph.add_node("data_extraction_tools", data_tools)
    graph.add_node("plotting_tools", plot_tools)
    graph.add_node("create_final_summary", create_final_summary)
    
    # Simplified graph flow for sequential processing demo
    graph.add_edge(START, "data_extraction_llm")
    graph.add_edge("data_extraction_llm", "plotting_llm")
    graph.add_edge("plotting_llm", "create_final_summary")
    graph.add_edge("create_final_summary", END)

    # In a full LangGraph setup, you'd use conditional edges:
    # graph.add_conditional_edges("data_extraction_llm", route_tools_or_next, {"tools": "data_extraction_tools", END: "plotting_llm"})
    # graph.add_edge("data_extraction_tools", "data_extraction_llm")
    
    compiled_graph = graph.compile()
    return compiled_graph

# =============================================================================
# STREAMLIT APP
# =============================================================================

def app():
    st.set_page_config(layout="wide", page_title="HuggingFace Agent Financial Analysis")
    st.title("🤖 Dual-Agent Financial Analysis (Hugging Face Local LLM)")
    st.caption(f"Using **CTransformers** with GGUF model: `{LOCAL_MODEL_PATH}`")
    st.warning("🚨 **Setup Note:** This requires a GGUF model file to be present at the specified path.")

    # Initialize session state for analysis results
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    # Load the graph once
    graph = build_dual_agent_graph()
    
    if graph is None:
        st.stop()

    user_input = st.text_input(
        "Enter your financial analysis request:",
        placeholder="e.g., Get daily stock data for TCS and plot with EMA overlays"
    )

    if st.button("Run Analysis", use_container_width=True):
        if not user_input:
            st.error("Please enter a request.")
            return

        with st.spinner("Running dual-agent analysis..."):
            try:
                # Clear previous state data
                DF_REGISTRY.clear()
                
                # Execute dual-agent analysis
                result = graph.invoke({
                    "messages": [{"role": "user", "content": user_input}],
                    "extracted_data_ids": [],
                    "plot_paths": []
                })
                
                st.session_state.analysis_result = result
            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {e}")
                logger.error(f"Execution Error: {e}")

    # Display Results
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        st.markdown(result.get("final_summary", "No summary available"))

        # Display Plot Image (Mock: check if a file was created and display)
        plot_paths = result.get("plot_paths", [])
        if plot_paths:
            st.header("🖼️ Visualization")
            plot_file_path = plot_paths[0].replace("file://", "").replace("/content/temp_plots", os.path.join(os.getcwd(), "temp_plots"))
            
            if os.path.exists(plot_file_path):
                st.image(plot_file_path, caption="Generated Plot/Dashboard")
            else:
                st.info(f"Mock plot path generated, but file not found (expected at: `{plot_file_path}`). Run the code locally to verify image creation.")

        st.subheader("💡 Agent Thoughts (Raw Output)")
        st.markdown(f"**Data Agent:** `{result.get('data_extraction_thought', 'N/A')}`")
        st.markdown(f"**Plotting Agent:** `{result.get('plotting_thought', 'N/A')}`")

if __name__ == "__main__":
    app()