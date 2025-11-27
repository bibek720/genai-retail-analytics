# GenAI Retail Analytics Assistant

A Multi-Agent AI system designed to analyze large-scale retail data, generate executive summaries, and answer business questions using Natural Language.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![GenAI](https://img.shields.io/badge/AI-Gemini%202.5-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

## Features
* **Conversational Q&A:** Chat with your data (e.g., "Which region has the highest cancellation rate?").
* **Executive Summaries:** Auto-generate KPIs and risk reports.
* **Multi-Agent Architecture:** Uses `LangGraph` for routing, code generation, and result validation.
* **Scalable Design:** Architecture blueprint included for handling 100GB+ datasets using Lakehouse patterns.

## Tech Stack
* **LLM:** Google Gemini 2.5 Flash (Optimized for low-latency analytics).
* **Orchestration:** LangGraph & LangChain.
* **Frontend:** Streamlit.
* **Data Engine:** Pandas (Prototype).


### 1. Steps to run the application
```bash
$ git clone <repo-link>
$ cd genai-retail-analytics
$ pip install -r requirements.txt
$ python3 process_data.py
$ streamlit run app.py
