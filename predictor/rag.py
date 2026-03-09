import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

def load_knowledge():

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "knowledge_base", "heart_disease.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )

    docs = splitter.split_text(text)

    documents = [Document(page_content=d) for d in docs]

    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(documents, embeddings)

    return db


db = load_knowledge()


def get_answer(question):

    docs = db.similarity_search(question)

    context = docs[0].page_content

    answer = f"Based on medical knowledge: {context}"

    return answer