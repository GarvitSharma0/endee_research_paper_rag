# 🧠 Research Paper RAG System using Endee Vector Database

---

## 🚀 Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that enables intelligent question-answering over research papers using:

- 🔍 **Endee Vector Database**
- 🤖 **Ollama (Mistral LLM)**
- 📄 **LangChain**
- ⚡ **FastAPI**

The system performs semantic search over research documents and generates context-aware answers using a local Large Language Model.

---

## 🎯 Problem Statement

Research papers are long and complex, making it difficult to quickly extract specific information.

This project solves that by:

1. Splitting research papers into semantic chunks  
2. Converting text into vector embeddings  
3. Storing embeddings in **Endee Vector Database**  
4. Retrieving the most relevant chunks  
5. Generating accurate answers using an LLM  

---

## 🏗️ System Architecture


PDF Research Papers
↓
Text Splitting
↓
Embeddings (MiniLM)
↓
Endee Vector Database
↓
Similarity Search (Top-K Retrieval)
↓
Mistral LLM (Ollama)
↓
Generated Answer


---

## 🛠️ Tech Stack

| Component        | Technology |
|------------------|------------|
| Backend API      | FastAPI |
| Vector Database  | Endee |
| Embeddings Model | sentence-transformers/all-MiniLM-L6-v2 |
| LLM              | Ollama (Mistral) |
| Framework        | LangChain |

---

## 🗄️ How Endee Is Used

Endee acts as the **core semantic retrieval engine** in this project.

### 🔹 Index Creation

```python
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
🔹 Storing Embeddings

Each document chunk is embedded using MiniLM

The vector is stored inside Endee

Metadata stores original text

🔹 Similarity Search

Query is embedded

Top-K most similar vectors retrieved

Retrieved text passed to LLM

Endee enables fast and scalable semantic search for this RAG system.

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/GarvitSharma0/endee_research_paper_rag.git
cd endee_research_paper_rag
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Install Ollama

Download from:
https://ollama.com

Pull the required model:

ollama pull mistral
4️⃣ Add Research Papers

Place your PDF files inside:

/papers
5️⃣ Run the Application
python -m uvicorn app:app --reload

Open in browser:

http://127.0.0.1:8000/docs

Use the /ask endpoint to query your research papers.

🧪 Example API Request
{
  "question": "What is the main contribution of this research paper?"
}
🔥 Key Features

✅ Endee-powered vector search

✅ Local LLM-based answer generation

✅ Retrieval-Augmented Generation (RAG)

✅ FastAPI REST interface

✅ Fully reproducible setup

📈 Future Improvements

Add frontend UI

Add multi-document filtering

Add agentic workflows

Deploy on cloud infrastructure

Add evaluation metrics for retrieval quality

👨‍💻 Author

Garvit Sharma
