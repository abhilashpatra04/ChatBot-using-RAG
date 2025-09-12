import os
import re
from typing import List, Dict, Any

import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.generativeai as genai

from app.config import settings
from app.rag import VectorStore, split_into_chunks

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/documents.readonly',
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/presentations.readonly',
]

MIMES = {
    'docs': 'application/vnd.google-apps.document',
    'sheets': 'application/vnd.google-apps.spreadsheet',
    'slides': 'application/vnd.google-apps.presentation',
    'pdf': 'application/pdf',
}

st.set_page_config(page_title="RAG Chatbot - Streamlit", page_icon="🤖", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'credentials' not in st.session_state:
    st.session_state.credentials = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'available_files' not in st.session_state:
    st.session_state.available_files = []
if 'model_key' not in st.session_state:
    st.session_state.model_key = 'gemini'


def _flow():
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(client_config=client_config, scopes=SCOPES)
    flow.redirect_uri = settings.google_oauth_redirect or "http://localhost:8501"
    return flow


def _drive(creds):
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def _docs(creds):
    return build('docs', 'v1', credentials=creds, cache_discovery=False)


def _sheets(creds):
    return build('sheets', 'v4', credentials=creds, cache_discovery=False)


def _slides(creds):
    return build('slides', 'v1', credentials=creds, cache_discovery=False)


def _list_files(creds, file_types: List[str]):
    drv = _drive(creds)
    files = []
    for ft in file_types:
        mime = MIMES[ft]
        try:
            res = drv.files().list(q=f"mimeType='{mime}' and trashed=false", pageSize=50, fields="files(id,name,modifiedTime,webViewLink,mimeType)", orderBy="modifiedTime desc").execute()
            for f in res.get('files', []):
                f['type'] = ft
                files.append(f)
        except Exception:
            continue
    files.sort(key=lambda f: f.get('modifiedTime',''), reverse=True)
    return files


def _extract(creds, file_id: str, file_type: str) -> str:
    if file_type == 'docs':
        d = _docs(creds).documents().get(documentId=file_id).execute()
        parts = []
        for elem in d.get('body', {}).get('content', []):
            p = elem.get('paragraph')
            if not p:
                continue
            for e in p.get('elements', []):
                t = e.get('textRun', {}).get('content')
                if t:
                    parts.append(t)
        return ''.join(parts)
    if file_type == 'sheets':
        svc = _sheets(creds)
        meta = svc.spreadsheets().get(spreadsheetId=file_id).execute()
        out = []
        for sht in meta.get('sheets', [])[:5]:
            title = sht.get('properties', {}).get('title', 'Sheet')
            try:
                val = svc.spreadsheets().values().get(spreadsheetId=file_id, range=f"'{title}'!A1:Z1000").execute()
                vals = val.get('values', [])
                if vals:
                    out.append(f"\n--- Sheet: {title} ---\n")
                    for row in vals:
                        out.append("\t".join(str(c) for c in row) + "\n")
            except Exception:
                continue
        return ''.join(out)
    if file_type == 'slides':
        svc = _slides(creds)
        pres = svc.presentations().get(presentationId=file_id).execute()
        out = []
        for i, slide in enumerate(pres.get('slides', [])):
            out.append(f"\n--- Slide {i+1} ---\n")
            for el in slide.get('pageElements', []):
                shp = el.get('shape')
                if shp and 'text' in shp:
                    for t in shp['text'].get('textElements', []):
                        tr = t.get('textRun')
                        if tr and tr.get('content'):
                            out.append(tr['content'])
        return ''.join(out)
    if file_type == 'drive':  # PDFs
        # Try export to text first
        try:
            data = _drive(creds).files().export(fileId=file_id, mimeType='text/plain').execute()
            if isinstance(data, bytes):
                txt = data.decode('utf-8', errors='ignore')
                if txt.strip():
                    return txt
            if isinstance(data, dict) and 'body' in data:
                txt = str(data['body']).strip()
                if txt:
                    return txt
        except Exception:
            pass
        # Fallback download + pdfminer
        try:
            import io
            from googleapiclient.http import MediaIoBaseDownload
            from pdfminer.high_level import extract_text as pdf_extract_text
            buf = io.BytesIO()
            req = _drive(creds).files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            buf.seek(0)
            return pdf_extract_text(buf) or ''
        except Exception:
            return ''
    return ''


def _gemini_model():
    if not settings.gemini_api_key:
        st.error("GEMINI_API_KEY not configured")
        return None
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model or "gemini-1.5-flash")
    return model


def _call_llm(prompt: str, model_key: str) -> str:
    """Call selected LLM and return text. model_key in {'gemini','groq-openai','groq-qwen'}"""
    if model_key == 'gemini':
        model = _gemini_model()
        if not model:
            return None
        try:
            resp = model.generate_content(prompt)
            return getattr(resp, 'text', None) or (resp.candidates[0].content.parts[0].text if getattr(resp, 'candidates', None) else "")
        except Exception as e:
            st.warning(f"Gemini error: {e}")
            return None
    else:
        # Groq
        if not settings.groq_api_key:
            st.error("GROQ_API_KEY not configured")
            return None
        try:
            from groq import Groq
            client = Groq(api_key=settings.groq_api_key)
            groq_model = 'openai/gpt-oss-20b' if model_key == 'groq-openai' else 'qwen/qwen3-32b'
            res = client.chat.completions.create(
                model=groq_model,
                messages=[{"role":"user","content": prompt}],
                temperature=0.1
            )
            return res.choices[0].message.content
        except Exception as e:
            st.warning(f"Groq error: {e}")
            return None


st.title("🤖 RAG Chatbot (Streamlit)")

with st.sidebar:
    st.header("Auth")
    if not st.session_state.authenticated:
        if not settings.google_client_id or not settings.google_client_secret:
            st.error("Google OAuth env vars missing")
        else:
            f = _flow()
            url, _ = f.authorization_url(prompt='consent')
            st.markdown(f"[Click to authenticate]({url})")
            code = st.text_input("Paste auth code")
            if st.button("Authenticate") and code:
                try:
                    f.fetch_token(code=code)
                    st.session_state.credentials = f.credentials
                    st.session_state.authenticated = True
                    st.success("Authenticated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Auth failed: {e}")
    else:
        st.success("Authenticated with Google")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.credentials = None
            st.session_state.available_files = []
            st.rerun()

    st.divider()
    st.header("Model")
    model_label = st.selectbox(
        "Choose LLM",
        index=0 if st.session_state.model_key == 'gemini' else (1 if st.session_state.model_key == 'groq-openai' else 2),
        options=["Gemini (Google)", "Groq - openai/gpt-oss-20b", "Groq - qwen/qwen3-32b"],
    )
    st.session_state.model_key = 'gemini' if model_label.startswith('Gemini') else ('groq-openai' if 'gpt-oss-20b' in model_label else 'groq-qwen')

    st.divider()
    st.header("Knowledge Base")
    if st.button("Reset KB"):
        st.session_state.vector_store.reset()
        st.success("KB cleared")

if not st.session_state.authenticated:
    st.info("Authenticate to load your Google files.")
    st.stop()

# File type selection
st.subheader("Select file types")
st.file_uploader  # no-op to stabilize layout
file_types = st.multiselect("Types", ['docs','sheets','slides','pdf'], default=['docs'])
if st.button("Load Files") and file_types:
    with st.spinner("Loading files..."):
        st.session_state.available_files = _list_files(st.session_state.credentials, file_types)
    st.success(f"Loaded {len(st.session_state.available_files)} files")

# File selector
files = st.session_state.available_files
if files:
    display = [f"{f['name']} ({f['type']})" for f in files]
    sel = st.multiselect("Choose files to ingest", options=list(range(len(files))), format_func=lambda i: display[i])
    if st.button("Ingest selected") and sel:
        with st.spinner("Ingesting..."):
            added = 0
            for i in sel:
                f = files[i]
                text = _extract(st.session_state.credentials, f['id'], f['type'])
                chunks = split_into_chunks(text)
                st.session_state.vector_store.delete_by_doc(f['id'])
                st.session_state.vector_store.add_texts(f['id'], chunks, title=f['name'], doc_type=f['type'])
                added += len(chunks)
        st.success(f"Ingested {added} chunks")

st.header("Chat")
for m in st.session_state.chat_history:
    role = "You" if m['role'] == 'user' else 'Assistant'
    st.markdown(f"**{role}:**\n\n{m['content']}")

q = st.text_input("Ask a question")
if st.button("Send") and q:
    st.session_state.chat_history.append({'role': 'user', 'content': q})

    ql = q.lower().strip()
    # Summarization style
    if any(w in ql for w in ["summarize","summary","overview","key points"]) or ql in {"summarize","summary"}:
        from app.rag import VectorStore as VS
        merged_ctx, titles = st.session_state.vector_store.build_merged_context(q, top_n_per_doc=3, max_docs=6)
        if merged_ctx:
            prompt = (
                "Write a concise markdown summary merging information from multiple documents. "
                "Use document titles as headings and bullet points under each. Avoid duplication.\n\n"
                f"Documents: {', '.join(titles)}\n\nContent:\n{merged_ctx}"
            )
            ans = _call_llm(prompt, st.session_state.model_key)
            if not ans:
                ans = "I couldn't generate a summary."
            st.session_state.chat_history.append({'role':'assistant','content': ans})
        else:
            st.session_state.chat_history.append({'role':'assistant','content': 'No ingested content available to summarize.'})
        st.rerun()

    # Normal Q&A
    results = st.session_state.vector_store.search(q, k=8)
    if results:
        ctx = "\n\n".join(chunk.text for _, chunk in results[:6])
        prompt = (
            "Answer strictly using the provided context from the user's selected Google files. "
            "If the context does not contain the answer, explicitly say you could not find it in their documents, then provide a concise general answer.\n\n"
            f"Question: {q}\n\nContext:\n{ctx}"
        )
        ans = _call_llm(prompt, st.session_state.model_key)
        if not ans:
            ans = "I couldn't generate a response."
        st.session_state.chat_history.append({'role': 'assistant', 'content': ans})
    else:
        prompt = f"Provide a concise answer from general knowledge.\n\nQuestion: {q}"
        ans = _call_llm(prompt, st.session_state.model_key)
        if not ans:
            ans = "I couldn't generate a response."
        ans = "I could not find an answer to your question in your documents.\n\n" + ans
        st.session_state.chat_history.append({'role': 'assistant', 'content': ans})

    st.rerun()
