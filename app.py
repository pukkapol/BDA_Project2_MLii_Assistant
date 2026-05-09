import streamlit as st
import os
import sys


try:
    import langchain
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error(f"❌ CRITICAL ERROR: {e}")
    st.warning("Please ensure your requirements.txt is correctly named and contains 'langchain', 'langchain-openai', etc.")
    st.stop()


st.set_page_config(page_title="BDA_Project2_GroupNo15", layout="wide")
st.title("🤖 BDA_Project2_yourGroupNo")
st.markdown("### MLii Fund Smart Assistant")

st.sidebar.markdown("""
**Group Members:**
- ID: 6631501089 - Name: Pukkapol Kangthong
""")


api_key = st.sidebar.text_input("OpenAI API Key", type="password")


if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    

    dataset_content = """
    ขั้นตอนการขอรับทุน MLii:
    1. ยื่นข้อเสนอโครงการ และหนังสือรับรอง
    2. คณะกรรมการพิจารณา
    3. ทำสัญญารับทุน เบิกจ่ายงวดที่ 1 (50%) [cite: 103, 129]
    4. ส่งรายงานความก้าวหน้า เบิกงวดที่ 2 (30%) [cite: 104, 115]
    5. ส่งรายงานสมบูรณ์ เบิกงวดที่ 3 (20%) [cite: 105, 132]
    
    งบประมาณ: ไม่เกิน 50,000 บาท ต่อโครงการ [cite: 43]
    ค่าตอบแทนผู้วิจัย: รวมไม่เกิน 3,000 บาท [cite: 72, 107]
    """

    @st.cache_resource
    def build_rag():
        from langchain.text_splitter import CharacterTextSplitter
        from langchain.schema import Document
        
        # Create a document from the content
        docs = [Document(page_content=dataset_content)]
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    try:
        qa_chain = build_rag()
        st.success("✅ RAG Pipeline is active!")
        
        query = st.text_input("ถามคำถามเกี่ยวกับทุน MLii:")
        if query:
            with st.spinner("Processing..."):
                response = qa_chain.invoke(query)
                st.info(response["result"])
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Enter your API Key in the sidebar to begin.")
