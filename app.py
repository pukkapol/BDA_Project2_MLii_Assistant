import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


st.set_page_config(page_title="BDA_Project2_GroupNo15", layout="centered")

st.title("BDA_Project2_GroupNo15")
st.header("MLii Fund Smart Assistant") 

st.markdown("""
**Group Members:**
* **Student ID:** 6631501089 - **Name:** Pukkapol Kangthong

""")

# Sidebar for API Key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 3.2 & 3.3 RAG Pipeline
    @st.cache_resource
    def load_data():
        # Using the content from your MLii Q&A and process documents 
        loader = TextLoader("mlii_dataset.txt", encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever()
        )

    try:
        qa_chain = load_data()
        st.success("System is ready to answer questions about the MLii fund!")

        user_input = st.text_input("Ask a question (e.g., 'เงินทุนงวดที่ 1 เบิกได้กี่เปอร์เซ็นต์?'):")
        
        if user_input:
            with st.spinner("Searching..."):
                # Answer based on provided documents [cite: 103, 115, 132]
                response = qa_chain.invoke(user_input)
                st.write("### Answer:")
                st.info(response["result"])
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter your OpenAI API Key in the sidebar to start.")
