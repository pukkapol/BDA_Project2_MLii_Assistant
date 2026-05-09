import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="BDA_Project2_GroupNo15", page_icon="🤖")

st.title("BDA_Project2_yourGroupNo")
st.markdown("### MLii Fund Smart Assistant")
st.markdown("""
**Group Members:**
1. Student ID: 6631501089 - Name: Pukkapol Kangthong
---
""")

# Setup OpenAI API Key input
api_key = st.text_input("Enter your OpenAI API Key to start:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    @st.cache_resource
    def load_rag_pipeline():
        # Load Data
        loader = TextLoader("mlii_dataset.txt", encoding="utf-8")
        documents = loader.load()
        
        # Split Text
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Create Embeddings & VectorStore (RAG Core)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Create QA Chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        return qa_chain

    try:
        qa_chain = load_rag_pipeline()
        st.success("RAG Pipeline loaded successfully!")
        
        # User Interaction
        query = st.text_input("ถามคำถามเกี่ยวกับทุน MLii (Ask a question about the MLii fund):")
        if query:
            with st.spinner("กำลังค้นหาคำตอบ..."):
                result = qa_chain.invoke(query)
                st.markdown("**Answer:**")
                st.info(result['result'])
                
    except Exception as e:
        st.error(f"Error setting up RAG: {e}. Please check your API key.")
else:
    st.warning("Please enter your API Key to initialize the RAG system.")
