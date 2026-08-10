import os
from typing import Callable, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_DIR = "faiss_store"

#Create the Groq chat model used for answering questions.
def get_llm() -> ChatGroq:
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.4,
    )


def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings()


#Check and add https:// if not in url
def normalize_url(url: str) -> str:
    url = url.strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


#Strip, fix scheme, and drop empty entries from user-entered URLs.
def clean_urls(raw_urls: List[str]) -> List[str]:
    return [u for u in (normalize_url(u) for u in raw_urls) if u]


def build_and_save_index(
    urls: List[str],
    index_dir: str = FAISS_INDEX_DIR,
    status_callback: Optional[Callable[[str], None]] = None,
    embedder: Optional[HuggingFaceEmbeddings] = None,
) -> None:
    
    def report(msg: str) -> None:
        if status_callback:
            status_callback(msg)

    report("Loading the data...")
    loader = WebBaseLoader(urls)

    data = loader.load()

    report("Splitting data into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = text_splitter.split_documents(data)

    report("Embedding the chunks...")
    embedder = embedder or get_embedder()
    vector_store = FAISS.from_documents(documents=docs, embedding=embedder)

    report("Saving the embeddings to disk...")
    vector_store.save_local(index_dir)


def index_exists(index_dir: str = FAISS_INDEX_DIR) -> bool:
    return os.path.exists(index_dir)


def load_vectorstore(index_dir: str = FAISS_INDEX_DIR, embedder: Optional[HuggingFaceEmbeddings] = None) -> FAISS:
    embedder = embedder or get_embedder()
    return FAISS.load_local(
        index_dir,
        embedder,
        allow_dangerous_deserialization=True,
    )


def answer_question(
    query: str,
    index_dir: str = FAISS_INDEX_DIR,
    vectorstore: Optional[FAISS] = None,
    llm: Optional[ChatGroq] = None,
) -> Dict[str, str]:

    vectorstore = vectorstore or load_vectorstore(index_dir)
    llm = llm or get_llm()

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )

    result = chain.invoke({"question": query}, return_only_outputs=True)

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", ""),
        }