# Ask-The-URL

Ask-The-URL is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about one or more webpages. It extracts webpage content, creates vector embeddings using HuggingFace, stores them in a FAISS vector database, and generates answers using Groq's Llama model.

## Screenshot

![Application Screenshot](assets/Screenshot_of_UI.png)

## Features

- Process up to three webpage URLs.
- Extract webpage content automatically.
- Semantic search using FAISS.
- Question answering using RAG.
- Source attribution.
- Streamlit interface.

## Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS
- Groq API

## Installation

```bash
git clone https://github.com/AV032/Ask-The-URL.git
cd Ask-The-URL
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

Run the application:

```bash
streamlit run main.py
```

## Project Structure

```
Ask-The-URL/
├── assets/
├── backend.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
