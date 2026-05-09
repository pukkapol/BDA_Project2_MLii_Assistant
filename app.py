import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# --- Project Info ---
st.set_page_config(page_title="MLii_Assistant", layout="wide")
st.title("🤖 MLii Fund AI Search")
st.sidebar.markdown("**Group Members:**\n- Student ID: 6631501089\n- Name: Pukkapol Kangthong")

# --- Dataset from your uploaded files ---
# This matches the guidelines in the PDF and Q&A documents you provided.
dataset_content = """
การเบิกจ่ายเงินทุนวิจัยแบ่งเป็น 3 งวด[cite: 239, 240, 241, 242]:
1. งวดที่ 1: ร้อยละ 50 ของทุน ภายใน 30 วันหลังทำสัญญา [cite: 240]
2. งวดที่ 2: ร้อยละ 30 หลังส่งรายงานความก้าวหน้า (6 เดือน) [cite: 241, 250]
3. งวดที่ 3: ร้อยละ 20 หลังส่งรายงานวิจัยฉบับสมบูรณ์และผ่านการประเมิน [cite: 242, 270, 271]

ข้อมูลสำคัญ:
- งบประมาณ: ไม่เกิน 50,000 บาท ต่อโครงการ [cite: 180]
- ค่าตอบแทนผู้วิจัย: รวมไม่เกิน 3,000 บาท ต่อโครงการ [cite: 209, 244]
- คุณสมบัติ: เป็นพนักงานเต็มเวลาไม่น้อยกว่า 1 ปี [cite: 183]
"""

@st.cache_resource
def load_local_ai():
    # Local Thai-capable AI (No API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = [Document(page_content=dataset_content)]
    splits = text_splitter.split_documents(docs)
    return FAISS.from_documents(splits, embeddings)

try:
    with st.spinner("⏳ Loading Local AI..."):
        vectorstore = load_local_ai()
    st.success("✅ System Ready")

    query = st.text_input("ถามคำถามเกี่ยวกับทุน MLii:")
    if query:
        results = vectorstore.similarity_search(query, k=1)
        if results:
            st.info(results[0].page_content)
except Exception as e:
    st.error(f"Error: {e}")
