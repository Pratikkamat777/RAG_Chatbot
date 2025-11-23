# RAG Chatbot - Chat My PDF

An AI-powered chatbot to answer questions from PDFs using OpenAI GPT, LangChain, and FAISS vector search. Built with Streamlit frontend and FastAPI backend.

## How to Run
1. Activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run backend: `uvicorn api:app --reload --port 8000`
4. Run frontend: `streamlit run app.py`
5. Upload PDFs and ask questions!
