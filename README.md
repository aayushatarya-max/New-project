# Personal Memory Search Engine

An AI-powered local desktop application that allows users to store, index, and semantically search their personal files (text, PDFs, images, voice notes).

Instead of searching by filename, users can search naturally using natural language queries (e.g., *"Where did I save the machine learning notes?"*). The system uses embeddings, SQLite for metadata, and FAISS vector database to retrieve and rank the most semantically relevant file content.

---

## Features

- **Multi-format Support**: Upload and index `.txt`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.mp3`, `.wav`, and `.m4a`.
- **Text Extraction Pipeline**:
  - PDFs using **PyMuPDF**
  - Images using OCR with **Tesseract OCR**
  - Voice Notes using speech-to-text with **OpenAI Whisper**
- **Semantic Vector Search**: Generates sentence embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`) and matches relevance using a **FAISS** vector index.
- **Hybrid & Filtering Search**: Combines semantic results with keyword matching, date filters, file-type filters, and Top-K thresholds.
- **FastAPI Backend**: Loosely coupled and modular endpoints.
- **Streamlit UI**: Clean sidebar control and responsive search interface.

---

## Project Structure

```text
personal-memory-search/
├── backend/                  # FastAPI Web Backend
│   ├── api/                  # API endpoints and dependency injection
│   ├── database/             # SQLite connection management
│   ├── models/               # SQLAlchemy DB Models
│   ├── repositories/         # Database persistence layers
│   ├── services/             # Independent business logic services
│   │   ├── pdf_service.py
│   │   ├── image_service.py
│   │   ├── speech_service.py
│   │   ├── chunk_service.py
│   │   ├── embedding_service.py
│   │   └── search_service.py
│   └── utils/                # Utility scripts & configurations (logger, etc.)
│
├── frontend/                 # Streamlit UI Layer
│   └── app.py
│
├── uploads/                  # Raw file upload storage (local files)
├── vector_store/             # FAISS index storage
├── tests/                    # Pytest test suite
├── .env                      # Local configuration settings (ignored in git)
├── .env.example              # Configuration environment template
├── .gitignore                # Git exclusions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation & Prerequisites

### 1. External System Dependencies

This project relies on two system-level packages that must be installed on your operating system:

#### A. FFmpeg (For Voice Transcriptions via Whisper)
- **Windows**: Install via [Chocolatey](https://chocolatey.org/) (`choco install ffmpeg`) or download the build binaries from [FFmpeg](https://ffmpeg.org/download.html) and add the bin directory to your Windows System `PATH`.
- Verify installation: `ffmpeg -version` in PowerShell or CMD.

#### B. Tesseract OCR (For Image OCR)
- **Windows**: Download and install the Tesseract executable (e.g. from [UB Mannheim's Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki)).
- Note down the installation path (typically `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- Put this path in the `.env` file under the `TESSERACT_CMD` variable.

### 2. Python Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```
2. **Activate the virtual environment**:
   - **Windows PowerShell**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows CMD / Git Bash**:
     ```bash
     source .venv/Scripts/activate
     ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Variables Configuration

Copy `.env.example` to `.env` and adjust variables if needed:
```bash
cp .env.example .env
```
Ensure `TESSERACT_CMD` points to your local Tesseract executable.

---

## Running the Application

1. **Start the Backend API**:
   ```bash
   uvicorn backend.main:app --reload
   ```
2. **Start the Streamlit UI**:
   ```bash
   streamlit run frontend/app.py
   ```
