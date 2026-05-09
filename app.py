%%writefile app.py
import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# --- 3.1 & 3.4 Project Info ---
st.set_page_config(page_title="BDA_Project2_GroupNo15", layout="wide")
st.title("🤖 BDA_Project2_GroupNo15")
st.markdown("### MLii Fund AI Search (No API Key Required)")

st.sidebar.markdown("""
**Group Members:**
- Student ID: 6631501089 - Name: Pukkapol Kangthong
---
**Model Info:**
Using Local Multilingual-MiniLM (AI)
""")

# --- 3.2 & 3.3 Dataset & RAG Pipeline ---
# This is your MLii Dataset
dataset_content = """
ขั้นตอนการขอรับทุนสนับสนุนวิจัยเพื่อพัฒนาการเรียนรู้ สถาบันนวัตกรรมการเรียนรู้ฯ
1. ยื่นข้อเสนอโครงการ พร้อมหนังสือรับรองจากคณะกรรมการหน่วยงาน และ Concept design 
2. เสนอคณะกรรมการวิจัยพิจารณาข้อเสนอโครงการ
3. ทำสัญญารับเงินทุน และเบิกจ่ายเงินทุนงวดที่ 1 (ร้อยละ 50)
4. ส่งรายงานความก้าวหน้าวิจัยเมื่อครบ 6 เดือน เบิกจ่ายเงินทุนงวดที่ 2 (ร้อยละ 30)
5. ส่งรายงานวิจัยฉบับสมบูรณ์ ผ่านการประเมิน เบิกจ่ายเงินทุนงวดที่ 3 (ร้อยละ 20)

งบประมาณวิจัย: ตั้งตามจริง ไม่เกิน 50,000 บาท ต่อโครงการ
ค่าตอบแทน: เบิกได้รวมกันไม่เกิน 3,000 บาท ต่อโครงการ
คุณสมบัติ: เป็นพนักงานเต็มเวลาไม่น้อยกว่า 1 ปี ไม่อยู่ระหว่างถูกระงับทุน
"""

@st.cache_resource
def load_local_ai():
    # 1. Initialize local Thai-supporting AI Embeddings (No Key Needed)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # 2. Prepare documents
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = [Document(page_content=dataset_content)]
    splits = text_splitter.split_documents(docs)
    
    # 3. Create Local Vector Store
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

with st.spinner("🚀 Loading Local Thai AI Model... (This may take a minute on first run)"):
    vectorstore = load_local_ai()

st.success("✅ Local AI System Active!")

# --- UI Interaction ---
query = st.text_input("ถามคำถามเกี่ยวกับทุน MLii (เช่น 'เบิกเงินงวดแรกได้กี่เปอร์เซ็นต์'):")

if query:
    with st.spinner("AI is analyzing context..."):
        # Search the dataset for the most relevant answer using AI meaning
        results = vectorstore.similarity_search(query, k=1)
        
        if results:
            st.markdown("### 📢 AI Found this Answer:")
            st.info(results[0].page_content)
        else:
            st.warning("ไม่พบข้อมูลที่เกี่ยวข้อง")
