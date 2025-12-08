# Financial News RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** system for summarizing and analyzing Vietnamese financial news using **FastAPI**, **ChromaDB**, **LangChain**, **Groq**, and **HuggingFace Embeddings**.

---

## Overview

This project builds a financial news chatbot capable of:

* Summarizing financial and economic articles from VNExpress
* Analyzing market trends (stocks, macroeconomics, currencies, companies…)
* Providing **data-driven short-term estimates** (clearly marked as non-investment advice)
* Responding in **fluent, professional Vietnamese**
* Maintaining conversation context using session-based memory

The system has three main components:

1. **build_data.py** — preprocess data & build the ChromaDB vector store
2. **rag_handler.py** — implement RAG retrieval and conversational logic
3. **server.py** — expose an API using FastAPI

---

## System Architecture

```
Data → Chunking → Embeddings → ChromaDB
                      ↓
                   FastAPI 
                      ↓
                     Groq
                      ↓
               RAG Chatbot Output
```

---

## Project Structure

```
.
├── crawl_news.py
├── build_data.py
├── rag_handler.py
├── server.py
├── vnexpress_kinhdoanh.json     # dataset input
├── chroma_store/                # generated vector database
└── static/
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NPTIsMyName/financial-chatbot-rag
cd financial-chatbot-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=optional_if_using_endpoint
```

---

## Step 1 — Build the Vector Database

The system uses HuggingFace embedding model and recursive chunking.
Run:
```bash
python crawl_news.py
```
Then:
```bash
python build_data.py
```

This will generate:

* A `chroma_store/` directory
* All text chunks stored with stable hashed IDs

---

## Step 2 — Start the Server

Run:

```bash
uvicorn server:app --reload --port 8000
```


## Key Features

* Automated text chunking
* Local or API-based HuggingFace embeddings
* Persistent vector storage with ChromaDB
* Session-based conversation memory
* Answer formatting optimized for financial writing
* Basic REST API for frontend integration

---

## Example Query

Input:

```
"Giá vàng dạo gần đây có biến động gì ?"
```

The chatbot will:

* Retrieve relevant financial articles
* Produce a clear and concise summary
* Offer analysis on causes
* Provide **light predictive insights** (if supported by text)
