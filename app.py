import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

st.set_page_config(page_title="BDA_Project2_MLii_Assistant", layout="wide")
st.title("🤖 MLii Fund AI Search")
st.markdown("### สถาบันนวัตกรรมการเรียนรู้ (No API Key Required)")

st.sidebar.markdown("""
**Group Members:**
- Student ID: 6631501089 - Name: Pukkapol Kangthong
""")

dataset_content = """
การเบิกจ่ายเงินทุนวิจัยแบ่งเป็น 3 งวด:
1. งวดที่ 1: ร้อยละ 50 ของทุน ภายใน 30 วันหลังทำสัญญา [cite: 103, 129]
2. งวดที่ 2: ร้อยละ 30 หลังส่งรายงานความก้าวหน้า (6 เดือน) [cite: 104, 113, 115]
3. งวดที่ 3: ร้อยละ 20 หลังส่งรายงานวิจัยฉบับสมบูรณ์และผ่านการประเมิน [cite: 105, 132]

ข้อมูลเพิ่มเติม:
- งบประมาณ: ไม่เกิน 50,000 บาท ต่อโครงการ [cite: 43]
- ค่าตอบแทนผู้วิจัย: รวมไม่เกิน 3,000 บาท [cite: 72, 107]
- คุณสมบัติ: เป็นพนักงานเต็มเวลาไม่น้อยกว่า 1 ปี และไม่อยู่ระหว่างถูกระงับทุน [cite: 46, 47]
"""

@st.cache_resource
def load_local_ai():
    # Uses a local model that understands Thai perfectly (No API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = [Document(page_content=dataset_content)]
    splits = text_splitter.split_documents(docs)
    
    return FAISS.from_documents(splits, embeddings)

try:
    with st.spinner("⏳ กำลังโหลดโมเดล AI ภาษาไทย..."):
        vectorstore = load_local_ai()
    st.success("✅ ระบบ AI พร้อมใช้งานแล้ว!")

    query = st.text_input("ถามคำถามเกี่ยวกับทุน MLii:")
    if query:
        with st.spinner("กำลังค้นหาคำตอบ..."):
            # Local AI search for the best match in your documents
            results = vectorstore.similarity_search(query, k=1)
            if results:
                st.info(results[0].page_content)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")
