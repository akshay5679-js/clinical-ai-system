from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

DB_PATH = "predictor/rag/vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

def ask_rag(question):

    docs = db.similarity_search(question, k=3)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a medical AI assistant.

Use the following medical knowledge to answer clearly.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]