import json
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load your faqs.json file
with open("faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Format as documents
documents = [f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data]
docs = [Document(page_content=text) for text in documents]

# Split long text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Use local sentence-transformer embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create or load ChromaDB
vectordb = Chroma.from_documents(split_docs, embedding, persist_directory="db")
vectordb.persist()

# Chat loop
retriever = vectordb.as_retriever(search_type="similarity", k=1)

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    results = retriever.get_relevant_documents(query)
    if results:
        print("Bot:", results[0].page_content)
    else:
        print("Bot: Sorry, I couldn't find an answer.")
