# RAG Chatbot - Chat My PDF

An AI-powered chatbot to answer questions from PDFs using OpenAI GPT, LangChain, and FAISS vector search. Built with Streamlit frontend and FastAPI backend.

## How to Run
1. Activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Create a .env file in the project root:`OPENAI_API_KEY=your_openai_api_key_here`
4. Run backend: `uvicorn api:app --reload --port 8000`
5. Run frontend: `streamlit run app.py`
6. Upload PDFs and ask questions!

