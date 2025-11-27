import streamlit as st
import sys
import os
import pandas as pd

# Add project root to path so we can import our agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the Agent function (Ensure src/agents/graph.py exists)
try:
    from src.agents.graph import run_chat
except ImportError:
    st.error("⚠️ Backend Agent not found! Make sure 'src/agents/graph.py' exists.")
    st.stop()

# --- 1. UI Configuration ---
st.set_page_config(
    page_title="Retail GenAI Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .stChatFloatingInputContainer {bottom: 20px;}
    .reportview-container {background: #f5f5f5;}
</style>
""", unsafe_allow_html=True)

# --- 2. Sidebar: Controls & Summarization Mode ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=50)
    st.title("Retail Analyst AI")
    st.markdown("---")
    
    st.subheader("⚙️ Modes")
    mode = st.radio("Select Mode", ["Conversational Q&A", "Executive Summary"])
    
    st.markdown("---")
    st.info("💡 **Tip:** Connects to Gemini 2.5 Flash for sub-second analytics.")
    
    # Reset History Button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 3. Main Interface Logic ---

st.title("🛍️ Interactive Retail Data Assistant")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have access to your Sales, Inventory, and Financial data. Ask me anything!"}
    ]

# --- MODE A: Executive Summary ---
if mode == "Executive Summary":
    st.header("📊 Executive Performance Report")
    
    if st.button("Generate Live Summary"):
        with st.spinner("Crunching numbers across 100GB dataset..."):
            # We trigger the agent specifically for a summary
            summary_prompt = "Generate a comprehensive executive summary of our sales performance, top categories, and risks."
            response = run_chat(summary_prompt)
            
            st.markdown("### 📝 AI-Generated Report")
            st.markdown(response)
            
            # Optional: Show a quick chart if the response mentions specific data
            # (In a real app, the agent would return the dataframe to plot)
            st.success("Report generated successfully using Real-Time RAG.")

# --- MODE B: Conversational Q&A ---
else:
    # Display historical messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ex: 'Which region has the highest cancellation rate?'"):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing data..."):
                try:
                    # CALL THE BACKEND AGENT
                    full_response = run_chat(prompt)
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"❌ Error: {str(e)}"
                    message_placeholder.error(full_response)
            
            # 3. Save Response
            st.session_state.messages.append({"role": "assistant", "content": full_response})