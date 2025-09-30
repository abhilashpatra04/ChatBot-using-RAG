# RAG Chatbot with Google Workspace Integration

A comprehensive Retrieval-Augmented Generation (RAG) chatbot that integrates with Google Workspace to provide intelligent document-based question answering. The system supports multiple file types from Google Drive, Sheets, Slides, and PDFs, with dual UI options (FastAPI web interface and Streamlit app).

## 🚀 Key Features

### 🔐 Authentication & Authorization
- **Google OAuth 2.0 Integration**: Secure authentication with Google Workspace
- **Session Management**: Persistent user sessions with secure cookie handling
- **Multi-Scope Access**: Read-only access to Drive, Docs, Sheets, and Presentations

### 📄 Multi-Format Document Support
- **Google Docs**: Native text extraction from Google Documents
- **Google Sheets**: Tabular data extraction with sheet-by-sheet processing
- **Google Slides**: Presentation content extraction from slide text elements
- **PDF Files**: Advanced PDF parsing using pdfminer.six for Google Drive PDFs
- **Batch Processing**: Support for ingesting multiple files simultaneously

### 🧠 Advanced RAG Pipeline

#### What is RAG (Retrieval-Augmented Generation)?
RAG is a technique that combines the power of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG:

1. **Retrieves** relevant information from a knowledge base (your documents)
2. **Augments** the user's question with this retrieved context
3. **Generates** a response using both the context and the LLM's capabilities

#### How RAG Works in This Project

**1. Document Ingestion & Processing:**
```
Google Files → Text Extraction → Chunking → Vector Embeddings → ChromaDB Storage
```

- **Text Extraction**: Each file type uses specialized extraction methods
- **Intelligent Chunking**: Documents are split into 1000-character chunks with 200-character overlap
- **Vector Embeddings**: Uses SentenceTransformer (`all-MiniLM-L6-v2`) for semantic understanding
- **Persistent Storage**: ChromaDB stores embeddings with metadata (title, doc_type, chunk_index)

**2. Query Processing:**
```
User Question → Vector Search → Context Retrieval → LLM Generation → Response
```

- **Semantic Search**: Converts questions to embeddings and finds similar document chunks
- **Diversified Retrieval**: Ensures results come from multiple documents (up to 4 docs, 3 chunks each)
- **Context Building**: Merges relevant chunks into coherent context
- **Smart Fallback**: Falls back to general knowledge when no relevant context is found

**3. Specialized Query Handling:**
- **Summarization Queries**: Detects summary requests and uses merged context across all documents
- **Q&A Queries**: Uses focused retrieval for specific questions
- **Source Attribution**: Always cites which documents were used for answers

### 🤖 Multiple LLM Support
- **Google Gemini**: Primary LLM with configurable models (gemini-1.5-pro, gemini-2.5-flash)
- **Groq Integration**: Alternative LLM providers with multiple models:
  - `openai/gpt-oss-20b` (OpenAI-compatible)
  - `qwen/qwen3-32b` (Qwen model)
- **Model Selection**: Users can choose between different LLMs in the UI

### 🎨 Dual User Interfaces

#### 1. FastAPI Web Interface (`/static/index.html`)
- **Modern Web UI**: Clean, responsive, dark-themed design with smooth interactions
- **File Management**: Collapsible type groups, per-group counts, Select‑all, quick links
- **Real-time Chat**: Markdown rendering, Enter-to-send, typing indicator
- **Model Selection**: Dropdown to choose between available LLMs
- **Web Search**: “Web search only” toggle and automatic fallback if docs lack answers
- **Source Badges**: Clickable hostnames for cited sources
- **Chat History**: Persistent conversation history

#### 2. Streamlit Application (`streamlit_app.py`)
- **Interactive Dashboard**: Full-featured Streamlit interface
- **Sidebar Controls**: Authentication, model selection, and knowledge base management
- **File Type Filtering**: Multi-select for different Google Workspace file types
- **Progress Indicators**: Visual feedback during file loading and ingestion
- **Session State Management**: Maintains state across interactions

### 🔧 Advanced Features

#### Vector Store Management
- **ChromaDB Integration**: Persistent vector database with automatic collection management
- **Document Versioning**: Automatic replacement of existing documents on re-ingestion
- **Debug Endpoints**: Admin endpoints for vector store inspection and debugging
- **Reset Functionality**: Complete knowledge base reset capability

#### Smart Context Handling
- **Overlap Management**: 200-character overlap between chunks prevents information loss
- **Sentence-Aware Splitting**: Chunks respect sentence boundaries for better coherence
- **Metadata Preservation**: Maintains document titles, types, and chunk relationships
- **Source Tracking**: Tracks which documents contributed to each answer

#### Error Handling & Fallbacks
- **Graceful Degradation**: Falls back to general knowledge when document context is insufficient
- **API Error Handling**: Robust error handling for Google APIs and LLM providers
- **Authentication Recovery**: Automatic token refresh for expired Google credentials
- **Empty Content Handling**: Skips empty or invalid documents during ingestion

## 📁 Project Structure

```
Codemate Assignment/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── auth.py                  # Google OAuth 2.0 authentication
│   ├── config.py                # Configuration management with Pydantic
│   ├── google_docs.py           # Google Workspace API integration
│   ├── main.py                  # FastAPI application entry point
│   ├── rag.py                   # RAG pipeline and vector store implementation
│   ├── routes.py                # API routes and chat logic
│   └── server.py                # Server configuration and startup
├── chroma_db/                   # ChromaDB vector database storage
│   ├── chroma.sqlite3          # SQLite database for ChromaDB metadata
│   └── [uuid folders]/         # Vector embeddings storage
├── static/                      # Static web assets
│   └── index.html              # FastAPI web interface
├── streamlit_app.py            # Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                  # This documentation
```

### Core Components

#### `app/rag.py` - RAG Pipeline Core
- **VectorStore Class**: Manages ChromaDB operations and vector embeddings
- **DocumentChunk Dataclass**: Represents processed document segments
- **Text Chunking**: Intelligent document splitting with overlap
- **Search Methods**: Semantic search, diversified search, and context building

#### `app/google_docs.py` - Google Workspace Integration
- **Multi-API Support**: Drive, Docs, Sheets, and Slides APIs
- **Text Extraction**: Specialized extraction for each file type
- **PDF Processing**: Advanced PDF parsing with fallback methods
- **File Listing**: Comprehensive file discovery and metadata retrieval

#### `app/auth.py` - Authentication System
- **OAuth 2.0 Flow**: Complete Google OAuth implementation
- **Session Management**: Secure session handling with URLSafeSerializer
- **Token Refresh**: Automatic credential refresh for expired tokens
- **Security**: HTTP-only cookies and CSRF protection

#### `app/routes.py` - API Logic
- **Chat Endpoints**: Question processing and response generation
- **Document Ingestion**: Batch file processing and vector storage
- **Admin Functions**: Debug endpoints and knowledge base management
- **LLM Integration**: Multi-provider LLM calling with error handling

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Google Cloud Platform account
- Google AI Studio account (for Gemini API)
- Optional: Groq account (for alternative LLM)

### 1. Google Cloud Console Setup

1. **Create a new project** or select existing one in [Google Cloud Console](https://console.cloud.google.com/)

2. **Enable required APIs**:
   - Google Drive API
   - Google Docs API
   - Google Sheets API
   - Google Slides API

3. **Create OAuth 2.0 credentials**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Application type: "Web application"
   - Authorized redirect URIs:
     - `http://localhost:8000/auth/callback` (for FastAPI)
     - `http://localhost:8501` (for Streamlit)

4. **Download credentials** and note your Client ID and Client Secret

### 2. API Keys Setup

#### Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Note the API key for configuration

#### Groq API Key (Optional)
1. Visit [Groq Console](https://console.groq.com/keys)
2. Create a new API key
3. Note the API key for configuration

### 3. Environment Configuration

1. **Copy the environment template**:
   ```bash
   cp ENV_TEMPLATE.txt .env
   ```

2. **Edit `.env` file** with your credentials:
   ```env
   # Google OAuth 2.0 credentials
   GOOGLE_CLIENT_ID=your_google_client_id_here
   GOOGLE_CLIENT_SECRET=your_google_client_secret_here
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/callback
   
   # Session secret (generate a random string)
   SESSION_SECRET=your_random_session_secret_here
   
   # Gemini API Key
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-1.5-pro
   
   # Groq API Key (Optional)
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```

### 4. Installation

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 5. Running the Application

#### Option 1: FastAPI Web Interface
```bash
uvicorn app.server:app --reload --port 8000
```
- Open browser to: `http://localhost:8000`
- Features: Modern web UI, real-time chat, file management

#### Option 2: Streamlit Application
```bash
streamlit run streamlit_app.py --server.port 8501
```
- Open browser to: `http://localhost:8501`
- Features: Interactive dashboard, advanced controls, session management

### 6. Usage Workflow

1. **Authentication**: Click "Sign in with Google" and complete OAuth flow
2. **Load Files**: Click "Load My Files" to discover your Google Workspace files
3. **Select Documents**: Choose files from different categories (Docs, Sheets, Slides, PDFs)
4. **Ingest Knowledge**: Click "Add to Knowledge Base" to process and store documents
5. **Start Chatting**: Ask questions about your documents or request summaries
6. **Model Selection**: Choose between Gemini, Groq-OpenAI, or Groq-Qwen models
7. **Web Search**:
   - Keep toggle OFF to use your documents first; the app falls back to the web if needed
   - Turn toggle ON to fetch answers strictly from the web (ignores ingested docs)

## 🔍 API Endpoints

### Authentication
- `GET /auth/login` - Initiate Google OAuth flow
- `GET /auth/callback` - OAuth callback handler
- `POST /auth/logout` - End user session

### Document Management
- `GET /docs/list` - List available Google Workspace files
- `POST /docs/ingest` - Ingest selected files into knowledge base

### Chat & RAG
- `POST /chat/ask` - Ask questions and get RAG-powered responses
- `GET /chat/history` - Retrieve chat history
- `POST /chat/history` - Add messages to chat history

### Administration
- `POST /admin/reset` - Reset vector store and chat history
- `GET /admin/debug` - Debug vector store contents

## 🚨 Important Notes

### Security Considerations
- **Production Deployment**: Use HTTPS, secure session secrets, and proper domain configuration
- **API Key Management**: Never commit API keys to version control
- **User Permissions**: The app requests read-only access to Google Workspace files
- **Session Security**: Sessions are stored in HTTP-only cookies with CSRF protection

### Performance & Scalability
- **Vector Store**: Currently uses in-memory ChromaDB; consider persistent storage for production
- **Chunking Strategy**: 1000-character chunks with 200-character overlap optimize for most use cases
- **Rate Limiting**: Google APIs have rate limits; consider implementing request throttling
- **Memory Usage**: Large document collections may require memory optimization
 - **Web Search**: HTML pages are fetched over HTTPS; network egress must be allowed. For environments behind a proxy, set `HTTP_PROXY`/`HTTPS_PROXY`.

### Limitations
- **File Size**: Very large files may cause memory issues during processing
- **API Quotas**: Google Workspace APIs have daily quotas
- **Model Costs**: LLM API calls incur costs based on usage
- **Language Support**: Optimized for English text; other languages may have reduced accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Authentication Errors**:
- Verify Google OAuth credentials are correct
- Check redirect URI matches exactly
- Ensure required APIs are enabled in Google Cloud Console

**API Key Issues**:
- Verify API keys are valid and have sufficient quotas
- Check environment variables are loaded correctly
- Ensure API keys have proper permissions

**Document Ingestion Problems**:
- Verify file permissions in Google Workspace
- Check file formats are supported
- Monitor console logs for specific error messages

**Vector Store Issues**:
- Clear ChromaDB data if corrupted: delete `chroma_db/` folder
- Check disk space for vector storage
- Verify sentence-transformers model downloads correctly

For additional support, please open an issue in the repository.
