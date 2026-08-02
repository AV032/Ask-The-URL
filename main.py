import os
import streamlit as st
import pickle
import time

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

st.title("Ask-The-Ulr")
st.sidebar.title("Enter Urls")

urls=[]

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.8
)


for i in range(3):
    url=st.sidebar.text_input(f"Url {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process Urls")

file_path='faiss_store.pkl'

placeholder=st.empty()

if process_url_clicked:

    if url and url.startswith(('http://','https://')):
        url = 'https://'+url
        
    #data loading

    placeholder.text('Loading the Data!')
    loader=WebBaseLoader(urls)
    data=loader.load()
    
    #data spliting
    
    placeholder.text('Spliting data into Chunks!')
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs=text_splitter.split_documents(data)

    # Embedding and saving embedding to FAISS index

    placeholder.text('Embedding the Chunks!')
    embedder=HuggingFaceEmbeddings()

    vector_store_groq = FAISS.from_documents(documents=docs,embedding=embedder)
    
    #saving the faiss index to a pickle file

    placeholder.text('Saving the Embeddings into a File!')
    with open(file_path,'wb') as  file:
        pickle.dump(vector_store_groq,file)


query=placeholder.text_input("Ask Your Question About the Urls ")

if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as file2:
            vectorstore=pickle.load(file2)

            chain=RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever())
            
            result=chain ({'question':query},return_only_outputs=True)

            st.header("Answer")
            st.write(result['answer'])

            sources=result.get('sources', '')
            
            if sources:
                st.subheader("Sources: ")
                source_list=sources.split('\n')
                for source in source_list:
                    st.write(source)