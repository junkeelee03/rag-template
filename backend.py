from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import streamlit as st

MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "vector_store_database"
COLLECTION_NAME = "embeddings_stream"
ATLAS_VECTOR_SEARCH = "vector_index_ghw"

def get_vector_store():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    embeddings = GoogleGenerativeAIEmbeddings(model="model/embeddings-001")
    vector_store = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings, index_name=ATLAS_VECTOR_SEARCH)
    return vector_store

def ingest_text(text_content):
    vector_store = get_vector_store()
    docs = Document(text_content)
    vector_store.add_documents([docs])
    return True

def get_rag_response(query):
    vector_store = get_vector_store()
    # might be wrong
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """"Use the following context from the user in order to provide
    an accurate anwer"""

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_stuff='stuff', retriever=retriver)

    response = qa_chain.invoke({"query": query})
    return response