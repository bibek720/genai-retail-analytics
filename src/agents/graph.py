import pandas as pd
import json
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. Define State ---
class AgentState(TypedDict):
    question: str
    intent: str
    query_logic: str 
    data_result: str
    final_answer: str

# --- 2. Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- 3. Define Agents ---

def query_resolution_agent(state: AgentState):
    """
    Agent 1: Intent Classification
    """
    print(f"🕵️  [Resolver] Analyzing: {state['question']}")
    
    prompt = f"""
    You are a Retail Analytics Router. 
    Analyze the user request: "{state['question']}"
    
    Output JSON only:
    {{
        "intent": "summarize" OR "qa",
        "parameters": "Any specific filters"
    }}
    """
    response = llm.invoke(prompt)
    try:
        parsed = json.loads(response.content.replace('```json', '').replace('```', ''))
    except:
        parsed = {"intent": "qa"} 
    
    return {"intent": parsed['intent']}

def data_extraction_agent(state: AgentState):
    """
    Agent 2: Code Generator (Multi-Dataset Aware)
    """
    print(f"🤖 [Extractor] Generating code for intent: {state['intent']}")
    
    # 1. Define the Schema for the LLM
    # This tells the AI what tables it has access to.
    schema_context = """
    You have access to the following pandas DataFrames:
    
    1. df_amazon (Amazon Sales):
       Columns: [Order ID, Date, Status, SKU, Category, Size, Qty, Amount, ship-city, ship-state]
       
    2. df_intl (International Sales):
       Columns: [Date, Customer_ID, SKU, Qty, Amount, Channel]
       
    3. df_inventory (Current Stock):
       Columns: [SKU, Stock_Quantity, Category, Size, Color]
       
    4. df_finance (Expenses & Income):
       Columns: [Particular, Amount, Type] (Type is 'Income' or 'Expense')
       
    5. df_pricing (Product Costs & MRP):
       Columns: [SKU, Category, Cost_Price, Amazon_MRP, Final_MRP, Report_Month]
       
    6. df_warehouse (Logistics Costs):
       Columns: [Service_Type, Shiprocket_Price, Increff_Price]
    """
    
    prompt = f"""
    You are a Python Data Engineer.
    {schema_context}
    
    User Question: {state['question']}
    Intent: {state['intent']}
    
    Write ONLY valid python pandas code.
    - Use the dataframe names listed above (e.g., df_amazon, df_inventory).
    - If a join is needed, merge them on 'SKU'.
    - Assign the final answer (string or number) to variable 'result'.
    - Do NOT generate plots, just data.
    """
    
    code_resp = llm.invoke(prompt)
    code = code_resp.content.replace('```python', '').replace('```', '').strip()
    
    try:
        # 2. Load ALL DataFrames
        base_path = "../../data/processed/"
        df_amazon = pd.read_csv(os.path.join(base_path, "fact_sales_amazon.csv"))
        df_intl = pd.read_csv(os.path.join(base_path, "fact_sales_intl.csv"))
        df_inventory = pd.read_csv(os.path.join(base_path, "dim_inventory.csv"))
        df_finance = pd.read_csv(os.path.join(base_path, "fact_financials.csv"))
        df_pricing = pd.read_csv(os.path.join(base_path, "fact_product_pricing.csv"))
        df_warehouse = pd.read_csv(os.path.join(base_path, "dim_warehouse_pricing.csv"))
        
        # 3. Create Execution Sandbox
        # We pass all DFs into the 'local_vars' dictionary
        local_vars = {
            'pd': pd,
            'df_amazon': df_amazon,
            'df_intl': df_intl,
            'df_inventory': df_inventory,
            'df_finance': df_finance,
            'df_pricing': df_pricing,
            'df_warehouse': df_warehouse
        }
        
        # 4. Execute Generated Code
        exec(code, {}, local_vars)
        result = str(local_vars.get('result', "No result found in 'result' variable."))
        
    except Exception as e:
        result = f"Error executing code: {str(e)}\nGenerated Code:\n{code}"
        
    return {"query_logic": code, "data_result": result}

def validation_agent(state: AgentState):
    """
    Agent 3: Synthesis
    """
    prompt = f"""
    You are a Business Analyst.
    Question: {state['question']}
    Data Result: {state['data_result']}
    
    Provide a clear, professional answer. 
    If the result is an error, explain what might have gone wrong.
    """
    response = llm.invoke(prompt)
    return {"final_answer": response.content}

# --- 4. Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("resolve", query_resolution_agent)
workflow.add_node("extract", data_extraction_agent)
workflow.add_node("validate", validation_agent)

workflow.set_entry_point("resolve")
workflow.add_edge("resolve", "extract")
workflow.add_edge("extract", "validate")
workflow.add_edge("validate", END)

app = workflow.compile()

# Helper for UI
def run_chat(question: str):
    result = app.invoke({"question": question})
    return result["final_answer"]