import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# 📁 Load your stored DB
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="db", embedding_function=embedding)
retriever = vectordb.as_retriever(search_type="similarity", k=1)

# 📦 Create FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    results = retriever.get_relevant_documents(req.question)
    if results:
        return {"answer": results[0].page_content}
    else:
        return {"answer": "Sorry, I couldn't find an answer."}
