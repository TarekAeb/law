import sys
import os
import re
import sqlite3
import concurrent.futures

# Patch sqlite3 for ChromaDB compatibility
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

DB_PATH = "algerian_laws_rag.db"
CHROMA_DIR = "chroma_db"

# Pull API key from Streamlit Secrets (for Cloud Deployment)
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

# ---------------------------------------------------------
# Auto-Download Databases from Hugging Face if Missing
# ---------------------------------------------------------
from huggingface_hub import snapshot_download, hf_hub_download
import shutil

def is_db_valid():
    # Check if DB file exists and is larger than a few KB (an empty sqlite db is around 0-8KB)
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) < 15000:
        return False
    # Check if Chroma directory exists and has files inside
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return False
    return True

if not is_db_valid():
    @st.cache_resource
    def download_databases():
        try:
            # Clean up potentially broken empty files/folders first
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            if os.path.exists(CHROMA_DIR):
                shutil.rmtree(CHROMA_DIR)
                
            print("Downloading algerian_laws_rag.db...")
            hf_hub_download(repo_id="tarekAeb/algerian-legal-db", repo_type="dataset", filename="algerian_laws_rag.db", local_dir=".")
            print("Downloading chroma_db...")
            snapshot_download(repo_id="tarekAeb/algerian-legal-db", repo_type="dataset", allow_patterns="chroma_db/*", local_dir=".")
            print("Downloads complete!")
        except Exception as e:
            st.error(f"Error downloading from Hugging Face: {e}")
            print(f"Error downloading from Hugging Face: {e}")
            
    with st.spinner("Downloading legal databases (First run only, this might take a minute)..."):
        download_databases()
# ---------------------------------------------------------

def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))

def render_result_text(text):
    direction = "rtl" if is_arabic(text) else "ltr"
    alignment = "right" if direction == "rtl" else "left"
    # Wrapping with \n\n tells Streamlit to render the inner content as Markdown
    st.markdown(f"<div dir='{direction}' style='text-align: {alignment}; font-family: Tahoma, Arial, sans-serif;'>\n\n{text}\n\n</div>", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict) and "text" in part]
        return "".join(texts)
    return str(content)

def get_extended_context(chunk_text, full_text, buffer=400):
    if not full_text: 
        return chunk_text
    search_str = chunk_text[:150]
    start_idx = full_text.find(search_str)
    if start_idx == -1:
        return chunk_text
    end_idx = start_idx + len(chunk_text)
    ext_start = max(0, start_idx - buffer)
    ext_end = min(len(full_text), end_idx + buffer)
    return full_text[ext_start:ext_end]

def hybrid_search(query_str, vectorstore, k=5):
    vector_results = vectorstore.similarity_search_with_score(query_str, k=k)
    keyword_results = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        safe_query = query_str.replace('"', '""')
        cursor.execute(f"""
            SELECT source_name, title, content, url, date, bm25(legal_documents_fts) 
            FROM legal_documents_fts 
            WHERE legal_documents_fts MATCH '"{safe_query}"'
            ORDER BY bm25(legal_documents_fts) ASC
            LIMIT ?
        """, (k,))
        for row in cursor.fetchall():
            keyword_results.append({
                "source": row[0] or "Unknown Source",
                "title": row[1] or "No Title",
                "text": row[2] or "", 
                "url": row[3] or "#",
                "date": row[4] or "Unknown Date"
            })
        conn.close()
    except Exception:
        pass

    merged = []
    for res in keyword_results:
        short_text = res["text"][:1500] if len(res["text"]) > 1500 else res["text"]
        merged.append({
            "source": res["source"],
            "date": res["date"],
            "url": res["url"],
            "title": f"🔑 [Keyword Match] {res['title']}",
            "raw_text": short_text
        })
        
    for doc, score in vector_results:
        chunk_text = doc.page_content.strip()
        source_id = doc.metadata.get("source_id")
        extended_text = chunk_text
        if source_id:
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM legal_documents WHERE id = ?", (source_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    full_text = row[0]
                    extended_text = get_extended_context(chunk_text, full_text, buffer=400)
                conn.close()
            except Exception:
                pass
        merged.append({
            "source": doc.metadata.get("source_name", "Unknown Source"),
            "date": doc.metadata.get("date", "Unknown Date"),
            "url": doc.metadata.get("url", "#"),
            "title": f"🧠 [Semantic Match] {doc.metadata.get('title', 'No Title')}",
            "raw_text": extended_text
        })
        
    seen = set()
    final_results = []
    for m in merged:
        if m["raw_text"] not in seen:
            seen.add(m["raw_text"])
            final_results.append(m)
            if len(final_results) >= k:
                break
    return final_results

@st.cache_data(show_spinner=False)
def generate_final_synthesis(query, formatted_results):
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)
    context = ""
    for idx, doc in enumerate(formatted_results, 1):
        context += f"Source {idx} ({doc['source']} - {doc['date']}):\n{doc['formatted_text']}\n\n"
        
    prompt = f"""You are a prestigious Algerian Legal Expert.
The user has asked a legal question. You have been provided with several official Algerian legal texts retrieved from the database.
Read the texts carefully and synthesize a clear, accurate, and professional answer to the user's question.
If the retrieved texts do not contain the answer, explicitly and politely state that the information is missing from the current database. Do NOT invent laws.
Always answer in the same language as the user's question (Arabic, French, or English).
Cite your sources naturally using the Source name and Date provided.

User Question: {query}

Retrieved Legal Context:
{context}
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return extract_text(response.content).strip()
    except Exception as e:
        return f"Error generating synthesis: {str(e)}"


st.set_page_config(page_title="Algerian Legal Search Engine", layout="centered")
st.title("⚖️ Algerian Legal Database Search")
st.markdown("Search across thousands of Algerian official laws, decrees, and Supreme Court decisions. Powered by **Gemini**.")

vectorstore = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "synthesis" in message:
            st.markdown("### 💡 Legal Synthesis")
            render_result_text(message["synthesis"])
        if "results" in message and message["results"]:
            st.markdown("### 📚 Official Legal Sources")
            for i, res in enumerate(message["results"], 1):
                st.markdown(f"**{i}. {res['source']} ({res['date']})**\n*[{res['title']}]({res['url']})*")
                with st.expander("Show Formatted Legal Text"):
                    render_result_text(res['formatted_text'])

if prompt := st.chat_input("E.g. What are the rules for money laundering?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Searching for relevant legal documents..."):
        results = hybrid_search(prompt, vectorstore, k=4)
        
        formatted_results = []
        if results:
            for res in results:
                # Basic cleanup instead of expensive LLM formatting
                res["formatted_text"] = res["raw_text"].strip()
                formatted_results.append(res)
                
    if not formatted_results:
        content = "No relevant legal texts found for this query."
        synthesis_text = ""
    else:
        content = "Retrieval complete. Generating synthesis..."
        with st.spinner("Generating final legal synthesis..."):
            synthesis_text = generate_final_synthesis(prompt, formatted_results)

    with st.chat_message("assistant"):
        st.markdown("### 💡 Legal Synthesis")
        render_result_text(synthesis_text)
        
        st.markdown("### 📚 Official Legal Sources")
        for i, res in enumerate(formatted_results, 1):
            st.markdown(f"**{i}. {res['source']} ({res['date']})**\n*[{res['title']}]({res['url']})*")
            with st.expander("Show Formatted Legal Text"):
                render_result_text(res['formatted_text'])
                
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "",
        "synthesis": synthesis_text,
        "results": formatted_results
    })
