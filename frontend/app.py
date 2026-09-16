import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Personal Memory Search Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

# Inject Custom CSS for aesthetic improvements
st.markdown("""
<style>
    .result-card {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .result-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
        color: var(--text-color);
    }
    .result-meta {
        font-size: 12px;
        color: gray;
        margin-bottom: 12px;
    }
    .result-text {
        font-size: 14px;
        line-height: 1.5;
    }
    .highlight {
        background-color: rgba(255, 255, 0, 0.3);
        padding: 2px 4px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def fetch_files():
    try:
        response = requests.get(f"{API_URL}/files")
        if response.status_code == 200:
            return response.json()
        else:
            st.sidebar.error("Failed to fetch files.")
            return []
    except Exception as e:
        st.sidebar.error(f"Error connecting to backend: {e}")
        return []

def delete_file(file_id):
    try:
        response = requests.delete(f"{API_URL}/files/{file_id}")
        if response.status_code == 200:
            st.sidebar.success("File deleted successfully.")
            return True
        else:
            st.sidebar.error(f"Failed to delete file: {response.json().get('detail')}")
            return False
    except Exception as e:
        st.sidebar.error(f"Error connecting to backend: {e}")
        return False

# --- Sidebar: Control Panel ---
with st.sidebar:
    st.title("🧠 Memory Manager")
    
    st.header("Upload New File")
    uploaded_file = st.file_uploader(
        "Supported: txt, pdf, png, jpg, jpeg, mp3, wav, m4a",
        type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
        help="Upload a document, image, or voice note to index it into your personal memory."
    )
    
    if uploaded_file is not None:
        if st.button("Upload and Index", type="primary"):
            with st.spinner("Processing and Indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{API_URL}/upload", files=files)
                    if res.status_code == 201:
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                    else:
                        error_msg = res.json().get("detail", "Unknown Error")
                        st.error(f"Upload failed: {error_msg}")
                except Exception as e:
                    st.error(f"Error during upload: {e}")
    
    st.divider()
    
    st.header("Indexed Files")
    if st.button("Refresh File List"):
        st.session_state["files"] = fetch_files()
        
    if "files" not in st.session_state:
        st.session_state["files"] = fetch_files()
        
    files = st.session_state["files"]
    if not files:
        st.info("No files indexed yet.")
    else:
        for f in files:
            with st.expander(f"📄 {f['filename']} ({f['filetype']})"):
                st.caption(f"Size: {f['filesize']} bytes")
                st.caption(f"Tags: {', '.join(f['tags'])}")
                if st.button("Delete File", key=f"del_{f['id']}", help="Permanently delete and de-index this file"):
                    if delete_file(f['id']):
                        st.session_state["files"] = fetch_files()
                        st.rerun()

# --- Main Area: Search Interface ---
st.title("Search Your Personal Memory")
st.markdown("Use natural language to search across your documents, images, and voice notes.")

search_query = st.text_input("What are you looking for?", placeholder="e.g., Where did I save my machine learning notes?")

# Filters
with st.expander("Search Filters & Settings", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        search_type = st.selectbox("Search Type", options=["hybrid", "semantic", "keyword"], index=0)
    with col2:
        limit = st.slider("Max Results", min_value=1, max_value=50, value=10)
    with col3:
        file_type_filter = st.selectbox("File Type", options=["All", "txt", "pdf", "image", "audio"], index=0)
    
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Start Date", value=None)
    with col5:
        end_date = st.date_input("End Date", value=None)

if st.button("Search", type="primary", use_container_width=True) or search_query:
    if not search_query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching..."):
            params = {
                "query": search_query,
                "type": search_type,
                "limit": limit
            }
            if file_type_filter != "All":
                params["filetype"] = file_type_filter
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
                
            try:
                response = requests.get(f"{API_URL}/search", params=params)
                if response.status_code == 200:
                    results = response.json()
                    
                    if not results:
                        st.info("No matching results found in your memory.")
                    else:
                        st.subheader(f"Found {len(results)} matches")
                        for idx, result in enumerate(results):
                            score = result.get('score', 0)
                            file_meta = result.get('file', {})
                            filename = file_meta.get('filename', 'Unknown')
                            file_type = file_meta.get('filetype', 'Unknown')
                            page = result.get('page_number', '?')
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-title">📄 {filename} <span style="font-size: 14px; font-weight: normal; color: gray;">(Page {page})</span></div>
                                <div class="result-meta">
                                    Type: {file_type.upper()} • Score: {score:.4f} • Relevance: {'🔥' if score > 0.7 else '👍'}
                                </div>
                                <div class="result-text">
                                    "{result['chunk_text']}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"Search failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error connecting to search engine: {e}")
