from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .auth import router as auth_router, get_credentials
from .google_docs import router as docs_router, extract_text_from_google_doc, get_doc_title, extract_text_from_sheet, extract_text_from_slides, extract_text_from_pdf, MIMES
from .rag import VectorStore, split_into_chunks
from .config import settings
import os
import json
from typing import List

api = APIRouter()

# In-memory vector store for demo scope. In production use a DB-backed store.
VECTOR_STORE = VectorStore()

# In-memory chat history (persists until server restart)
CHAT_HISTORY = []

api.include_router(auth_router)
api.include_router(docs_router)


@api.post("/admin/reset")
def admin_reset():
    VECTOR_STORE.reset()
    CHAT_HISTORY.clear()
    return {"status": "vector store and chat history reset"}


@api.get("/chat/history")
def get_chat_history():
    return {"history": CHAT_HISTORY}


@api.post("/chat/history")
def add_to_chat_history(payload: dict):
    role = payload.get("role", "user")
    content = payload.get("content", "")
    if role and content:
        CHAT_HISTORY.append({"role": role, "content": content})
    return {"status": "added to history"}


@api.get("/admin/debug")
def debug_vector_store():
    """Debug endpoint to see what's in the vector store"""
    try:
        # Get all documents from the collection
        results = VECTOR_STORE.collection.get()
        docs = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        
        # Group by doc_type
        by_type = {}
        for doc, meta in zip(docs, metadatas):
            doc_type = meta.get("doc_type", "unknown")
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append({
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "chunk_length": len(doc)
            })
        
        return {
            "total_chunks": len(docs),
            "by_type": by_type,
            "sample_chunks": docs[:3] if docs else []
        }
    except Exception as e:
        return {"error": str(e)}


@api.post("/docs/ingest")
def ingest_docs(request: Request, payload: dict):
    creds = get_credentials(request)
    if not creds:
        return JSONResponse({"error": "not authenticated"}, status_code=401)
    items: List[dict] = payload.get("items") or []
    # backward compat: allow {doc_ids: []} for docs only
    doc_ids = payload.get("doc_ids", [])
    added = 0
    if items:
        for it in items:
            file_id = it.get("id")
            file_type = it.get("type", "docs")
            if not file_id:
                continue
            VECTOR_STORE.delete_by_doc(file_id)
            title = get_doc_title(creds, file_id)
            if file_type == "docs":
                text = extract_text_from_google_doc(creds, file_id)
            elif file_type == "sheets":
                text = extract_text_from_sheet(creds, file_id)
            elif file_type == "slides":
                text = extract_text_from_slides(creds, file_id)
            elif file_type == "drive":  # PDFs from Google Drive
                text = extract_text_from_pdf(creds, file_id)
            else:
                text = ""
            
            # Debug logging
            print(f"Processing {file_type} file {file_id} ({title}): extracted {len(text)} chars")
            
            chunks = split_into_chunks(text)
            if chunks:
                VECTOR_STORE.add_texts(file_id, chunks, title=title, doc_type=file_type)
                added += len(chunks)
                print(f"Added {len(chunks)} chunks for {file_type} file {file_id}")
            else:
                print(f"No chunks created for {file_type} file {file_id} - text was empty or too short")
    else:
        for file_id in doc_ids:
            VECTOR_STORE.delete_by_doc(file_id)
            title = get_doc_title(creds, file_id)
            text = extract_text_from_google_doc(creds, file_id)
            chunks = split_into_chunks(text)
            if chunks:
                VECTOR_STORE.add_texts(file_id, chunks, title=title, doc_type="docs")
                added += len(chunks)
    return {"status": f"ingested {added} chunks"}


def _is_summarize_query(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["summarize", "summary", "overview", "key points", "tl;dr"]) or ql.strip() in {"summarize", "summary"}


def _call_llm(prompt: str, selected_model: str, gemini_key: str, groq_key: str, gemini_model_name: str, groq_model_name: str) -> str:
    """Call the selected LLM with the given prompt"""
    if selected_model in ("groq-openai", "groq-qwen") and groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            effective_model = "openai/gpt-oss-20b" if selected_model == "groq-openai" else "qwen/qwen3-32b"
            response = client.chat.completions.create(
                model=effective_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq error: {repr(e)}")
            return None
    
    elif selected_model == "gemini" and gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(gemini_model_name)
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "")
            return text
        except Exception as e:
            print(f"Gemini error: {repr(e)}")
            return None
    
    return None


def _add_to_history_and_return(answer: str, source: str = "general", sources: list = None):
    """Add assistant response to history and return the response"""
    CHAT_HISTORY.append({"role": "assistant", "content": answer})
    return {"answer": answer, "source": source, "sources": sources or []}


@api.post("/chat/ask")
def ask(payload: dict):
    question = payload.get("question", "").strip()
    if not question:
        return {"answer": "Please enter a question."}

    # Add user question to history
    CHAT_HISTORY.append({"role": "user", "content": question})

    # Get model selection from payload, default to gemini
    selected_model = payload.get("model", "gemini")
    gemini_key = settings.gemini_api_key
    groq_key = settings.groq_api_key
    gemini_model_name = settings.gemini_model or "gemini-2.5-flash"
    groq_model_name = settings.groq_model or "openai/gpt-oss-20b"

    # Summarization path: merge context across docs
    if _is_summarize_query(question):
        merged_context, titles = VECTOR_STORE.build_merged_context(question, top_n_per_doc=3, max_docs=6)
        if merged_context:
            prompt = (
                "You are a helpful assistant. Write a concise, well-structured summary that synthesizes information from multiple documents.\n"
                "Use headings with document titles and bullet points. Avoid duplication.\n\n"
                f"Documents: {', '.join(titles)}\n\n"
                f"Content snippets from documents (mixed order):\n{merged_context}\n\n"
                "Output a short markdown summary."
            )
            text = _call_llm(prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
            if text:
                prefix = "From your documents (sources: " + ", ".join(titles[:3]) + "): " if titles else "From your documents: "
                return _add_to_history_and_return(prefix + text, "docs", titles)

    # Normal Q&A path
    retrieved = VECTOR_STORE.search(question, k=8)
    context_chunks = [c.text for _, c in retrieved]
    context_text = "\n\n".join(context_chunks[:6])
    sources = VECTOR_STORE.top_source_titles(question, k=6)
    
    # Debug logging
    print(f"Search results: {len(retrieved)} chunks found")
    for i, (score, chunk) in enumerate(retrieved[:3]):
        print(f"  {i+1}. {chunk.doc_type} - {chunk.title}: {chunk.text[:100]}...")

    if context_text:
        prompt = (
            "You are a helpful assistant. Answer Strictly using the provided context from the user's Google Docs selected Documents. "
            "If the context does not contain the answer, explicitly say you could not find it in their documents and do not fabricate.\n\n"
            f"Question: {question}\n\nContext from multiple documents (may include repeated info):\n{context_text}\n\nInstructions:\n- Cite up to 2 short snippets from the context in quotes.\n- Be concise.\n- Start with 'From your documents:' if grounded."
        )
        text = _call_llm(prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
        if text:
            lower = text.lower()
            if any(p in lower for p in ["could not find", "couldn't find", "not found", "do not have enough information"]):
                general_prompt = (
                    "Provide a concise, factual answer from general knowledge. Avoid fabrications and include a short explanation.\n\n"
                    f"Question: {question}"
                )
                general_text = _call_llm(general_prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
                if general_text:
                    text = text.rstrip() + "\n\n" + general_text
            prefix = "From your documents (sources: " + ", ".join(sources[:3]) + "): " if sources else "From your documents: "
            return _add_to_history_and_return(prefix + text, "docs", sources)

    # Fallback to general knowledge
    prefix = "I could not find an answer to your question in your documents. "
    prompt = (
        "Provide a concise, factual answer from general knowledge. Avoid fabrications and include a short explanation.\n\n"
        f"Question: {question}"
    )
    text = _call_llm(prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
    if text:
        return _add_to_history_and_return(prefix + text, "general", [])
    
    # Final fallback
    model_name = "Gemini" if selected_model == "gemini" else ("Groq-openai" if selected_model == "groq-openai" else "Groq-qwen")
    return _add_to_history_and_return(f"I could not find an answer to your question in your documents. ({model_name} error; check API key/model and internet)", "general", [])
