🧠 Research Paper RAG using Endee Vector Database
🚀 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions from research papers using:

🔍 Endee Vector Database

🤖 Ollama (Mistral LLM)

📄 LangChain for document processing

⚡ FastAPI backend

The system performs semantic search over research papers and generates context-aware answers using a local LLM.

🎯 Problem Statement

Large research papers are difficult to search and query efficiently.

This project solves that by:

Splitting research papers into semantic chunks

Generating embeddings

Storing them in Endee Vector DB

Retrieving relevant context

Generating AI-powered answers

🏗 System Architecture
PDF Papers
    ↓
Text Splitting
    ↓
Embeddings (MiniLM)
    ↓
Endee Vector Database
    ↓
Similarity Search
    ↓
Mistral LLM (Ollama)
    ↓
Generated Answer
🛠 Tech Stack
Component	Technology
Backend	FastAPI
Vector DB	Endee
Embeddings	sentence-transformers/all-MiniLM-L6-v2
LLM	Ollama (Mistral)
Framework	LangChain
🗄 How Endee Is Used

Endee is used as the core vector database:

Creating index with cosine similarity

Storing embedding vectors

Performing similarity search

Returning top-K relevant documents

Example usage in code:

from endee import Endee
from endee.schema import Schema
from endee.constants import SpaceType

db = Endee()

schema = Schema(
    name="research_index",
    dimension=384,
    space_type=SpaceType.COSINE
)

index = db.create_index(schema)

Endee powers the semantic retrieval layer of this RAG system.

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/GarvitSharma0/your-repo-name.git
cd your-repo-name
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Install Ollama

Download from:
https://ollama.com

Pull model:

ollama pull mistral
4️⃣ Add Research Papers

Place PDF files inside:

/papers
5️⃣ Run Application
python -m uvicorn app:app --reload

Open:

http://127.0.0.1:8000/docs

Use /ask endpoint to query papers.

🧪 Example Query
{
  "question": "What is the main contribution of this research paper?"
}
🔥 Key Features

✔ Endee vector search
✔ Local LLM generation
✔ Semantic similarity retrieval
✔ Clean FastAPI API
✔ Fully reproducible setup

📌 Future Improvements

Add UI frontend

Add multi-document filtering

Add agentic workflows

Deploy on cloud

👨‍💻 Author

Garvit Sharma
