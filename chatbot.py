# """this module contains the logic for chatbot"""

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_community.document_loaders.blob_loaders import Blob
import logging
from langchain_community.document_loaders.parsers import PyPDFParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


# # Embeddings + LLM
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Chroma v1 requires specifying a collection
chroma = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"   
)

retriever = chroma.as_retriever(search_kwargs={"k": 4})


# Prompt
TEMPLATE = """
<context>
{context}
</context>

<question>
{input}
</question>

- Answer from provided context
- If not found, reply: "I couldnt find an answer"
"""

PROMPT = ChatPromptTemplate.from_template(TEMPLATE)

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | PROMPT
    | llm
)


def store_document(documents: list[Document]):
    try:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        ids = chroma.add_texts(
            texts=texts,
            metadatas=metadatas
        )


        return {
            "status": "success",
            "total": len(ids),
            "documents": ids
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "documents": [],
            "total": 0
        }

# ------------ Retrieval ------------
def retrieve_document(query: str) -> list[Document]:
    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} docs for query: {query}")
    return docs


# ------------ Ask ------------
def ask_question(query: str):
    logger.info(f"Query: {query}")
    response = rag_chain.invoke(query)
    logger.info(f"LLM Response: {response}")
    return response


# ------------  PDF parse ------------
parser = PyPDFParser()

def parse_pdf(file_content: bytes) -> list[Document]:
    blob = Blob(data=file_content)
    docs = [doc for doc in parser.lazy_parse(blob)]
    logger.info(f"Parsed {len(docs)} pages from PDF")
    return docs

