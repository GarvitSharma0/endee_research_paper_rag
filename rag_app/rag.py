import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================
# STEP 1: LOAD PDF FILES
# =========================
def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())

    return documents


# =========================
# STEP 2: SPLIT DOCUMENTS
# =========================
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)


# =========================
# STEP 3: CREATE VECTOR DB
# =========================
def create_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="db"
    )

    return db


# =========================
# STEP 4: LOAD LLM
# =========================
def load_llm():
    return Ollama(model="mistral")


# =========================
# STEP 5: BUILD RAG CHAIN
# =========================
def build_rag_chain(db, llm):
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":

    print("🔄 Loading PDFs...")
    docs = load_pdfs("papers")

    print("✂️ Splitting documents...")
    chunks = split_docs(docs)

    print("🧠 Creating vector database...")
    db = create_vectorstore(chunks)

    print("🤖 Loading local LLM...")
    llm = load_llm()

    print("🔗 Building RAG system...")
    rag_chain = build_rag_chain(db, llm)

    print("\n✅ RAG system ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        try:
            result = rag_chain.invoke(query)
            print("AI:", result)
        except Exception as e:
            print("Error:", str(e))