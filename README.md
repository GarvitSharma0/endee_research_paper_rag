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
```
### 🔹 Storing Embeddings

- Each document chunk is embedded using **MiniLM**
- The generated vector is stored inside **Endee**
- Metadata stores the original text content for retrieval

---

### 🔹 Similarity Search

- The user query is embedded using the same embedding model
- Top-K most similar vectors are retrieved from Endee
- Retrieved text chunks are passed to the LLM
- The LLM generates a context-aware answer

Endee enables fast and scalable semantic search for this RAG system.

---

# ⚙️ Setup Instructions
## 1️⃣ Clone Repository
First, clone the project from GitHub and navigate into the project directory:

Bash
git clone [https://github.com/GarvitSharma0/endee_research_paper_rag.git](https://github.com/GarvitSharma0/endee_research_paper_rag.git)
cd endee_research_paper_rag
## 2️⃣ Initialize Endee Index
Use the following Python snippet to initialize your vector database:

Python
from endee import Endee
from endee.schema import Schema
from endee.constants import SpaceType

db = Endee()

# Initialize the schema for 384-dimension MiniLM embeddings
schema = Schema(
    name="research_index",
    dimension=384,
    space_type=SpaceType.COSINE
)

index = db.create_index(schema)
## 3️⃣ Install Dependencies
Install the required libraries:

Bash
pip install -r requirements.txt
## 4️⃣ Configure Local LLM
Ensure Ollama is installed, then pull the model:

Bash
ollama pull mistral
## 5️⃣ Run the Application
Start the FastAPI server:

Bash
python -m uvicorn app:app --reload
