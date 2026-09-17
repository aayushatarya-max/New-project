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

def get_file_content(file_id):
    try:
        res = requests.get(f"{API_URL}/files/{file_id}/download", timeout=15)
        if res.status_code == 200:
            return res.content
    except Exception:
        pass
    return None

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
                    res = requests.post(f"{API_URL}/upload", files=files, timeout=90)
                    if res.status_code == 201:
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                        st.session_state["files"] = fetch_files()
                        st.rerun()
                    else:
                        try:
                            error_msg = res.json().get("detail", "Unknown Error")
                        except Exception:
                            error_msg = f"Server returned HTTP status code {res.status_code}"
                        st.error(f"Upload failed: {error_msg}")
                except requests.exceptions.Timeout:
                    st.error("Upload timed out. Try uploading a smaller file or image.")
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
                
                if st.button("📁 Open File", key=f"sb_open_{f['id']}", use_container_width=True):
                    try:
                        open_res = requests.post(f"{API_URL}/files/{f['id']}/open-local", timeout=5)
                        if open_res.status_code == 200:
                            st.success(f"Opened '{f['filename']}' locally.")
                        else:
                            f_b = get_file_content(f['id'])
                            if f_b:
                                st.download_button("📥 Download File", data=f_b, file_name=f['filename'], key=f"sb_dl_{f['id']}", use_container_width=True)
                    except Exception:
                        pass

                if st.button("Delete File", key=f"del_{f['id']}", help="Permanently delete and de-index this file", use_container_width=True):
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
                            file_id = result.get('file_id') or result.get('file', {}).get('id')
                            filename = result.get('filename') or result.get('file', {}).get('filename', 'Unknown File')
                            file_type = (result.get('filetype') or result.get('file', {}).get('filetype', 'Unknown')).lower()
                            page = result.get('page_number', 1)
                            chunk_text = result.get('chunk_text', '')
                            
                            # Truncate long snippets to keep cards compact
                            snippet = chunk_text[:280] + "..." if len(chunk_text) > 280 else chunk_text

                            card_container = st.container()
                            with card_container:
                                col_info, col_btn = st.columns([3.5, 1])
                                with col_info:
                                    st.markdown(f"""
                                    <div style="padding: 10px; border: 1px solid #e0e0e0; border-radius: 6px; background-color: #f9f9f9; margin-bottom: 8px;">
                                        <div style="font-size: 16px; font-weight: 600; color: #111;">📄 {filename} <span style="font-size: 13px; font-weight: normal; color: #666;">(Page/Section {page})</span></div>
                                        <div style="font-size: 12px; color: #555; margin-top: 4px; margin-bottom: 6px;">
                                            <b>Type:</b> {file_type.upper()} &nbsp;|&nbsp; <b>Relevance:</b> {score:.2f} ({'🔥 High' if score > 0.6 else '👍 Relevant'})
                                        </div>
                                        <div style="font-size: 13px; color: #333; font-style: italic;">
                                            "{snippet}"
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                with col_btn:
                                    if file_id:
                                        # Open File action button
                                        if st.button("📁 Open File", key=f"open_file_{file_id}_{idx}", use_container_width=True):
                                            # 1. Attempt local Windows file opening via API
                                            try:
                                                open_res = requests.post(f"{API_URL}/files/{file_id}/open-local", timeout=5)
                                                if open_res.status_code == 200:
                                                    st.success(f"Opened '{filename}' locally.")
                                                else:
                                                    # 2. Live Cloud fallback
                                                    st.info(f"File stored on Cloud/Server. Click below to view.")
                                                    file_b = get_file_content(file_id)
                                                    if file_b:
                                                        st.download_button(
                                                            label=f"📥 Download {filename}",
                                                            data=file_b,
                                                            file_name=filename,
                                                            key=f"dl_fallback_{file_id}_{idx}",
                                                            use_container_width=True
                                                        )
                                            except Exception:
                                                file_b = get_file_content(file_id)
                                                if file_b:
                                                    st.download_button(
                                                        label=f"📥 Download {filename}",
                                                        data=file_b,
                                                        file_name=filename,
                                                        key=f"dl_exc_{file_id}_{idx}",
                                                        use_container_width=True
                                                    )
                else:
                    st.error(f"Search failed: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Error connecting to search engine: {e}")
