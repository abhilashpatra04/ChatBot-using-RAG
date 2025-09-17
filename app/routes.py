from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .auth import router as auth_router, get_credentials
from .google_docs import router as docs_router, extract_text_from_google_doc, get_doc_title, extract_text_from_sheet, extract_text_from_slides, extract_text_from_pdf, MIMES
from .rag import VectorStore, split_into_chunks
from .config import settings
import os
import json
from typing import List
import requests
import urllib.parse
import re

api = APIRouter()


VECTOR_STORE = VectorStore()


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

        results = VECTOR_STORE.collection.get()
        docs = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        

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


def _duckduckgo_search(query: str, max_results: int = 5) -> List[dict]:
    """Lightweight search via DuckDuckGo endpoints (no API key). Returns list of {title, url}."""
    headers = {"User-Agent": "Mozilla/5.0"}
    q = urllib.parse.quote_plus(query)
    endpoints = [
        f"https://duckduckgo.com/html/?q={q}",
        f"https://duckduckgo.com/lite/?q={q}",
    ]
    links: List[dict] = []
    for url in endpoints:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            html = resp.text or ""
            # Broadly capture anchors with http(s) hrefs and non-empty titles
            for m in re.finditer(r'<a[^>]+href=\"(https?[^\"]+)\"[^>]*>(.*?)<', html, re.IGNORECASE):
                href = m.group(1)
                netloc = urllib.parse.urlparse(href).netloc
                if not netloc or "duckduckgo.com" in netloc:
                    continue
                title = re.sub(r"<[^>]+>", "", m.group(2))
                title = re.sub(r"\s+", " ", title).strip()
                if not title:
                    continue
                links.append({"title": title, "url": href})
                if len(links) >= max_results:
                    return links
        except Exception:
            continue
    return links


def _bing_search(query: str, max_results: int = 5) -> List[dict]:
    """Fallback search via Bing HTML (no API key). Returns list of {title, url}."""
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    q = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={q}&setlang=en"
    links: List[dict] = []
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text or ""
        # Prefer standard result blocks
        for m in re.finditer(r'<li class=\"b_algo\"[\s\S]*?<h2>\s*<a[^>]+href=\"(https?[^\"]+)\"[^>]*>([\s\S]*?)<', html, re.IGNORECASE):
            href = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2))
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue
            links.append({"title": title, "url": href})
            if len(links) >= max_results:
                return links
        # Fallback: any anchor inside <h2>
        for m in re.finditer(r'<h2>\s*<a[^>]+href=\"(https?[^\"]+)\"[^>]*>([\s\S]*?)<', html, re.IGNORECASE):
            href = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2))
            title = re.sub(r"\s+", " ", title).strip()
            if not title:
                continue
            links.append({"title": title, "url": href})
            if len(links) >= max_results:
                return links
    except Exception:
        pass
    return links


def _ddg_instant(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo Instant Answer JSON API (no key). Returns {title,url}."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&no_redirect=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json() if resp.status_code == 200 else {}
        out: List[dict] = []
        if data.get("AbstractURL"):
            out.append({"title": data.get("Heading") or data.get("AbstractURL"), "url": data.get("AbstractURL")})
        for item in data.get("RelatedTopics", []) or []:
            if isinstance(item, dict):
                first_url = item.get("FirstURL")
                text = item.get("Text") or first_url
                if first_url and first_url.startswith("http"):
                    out.append({"title": text, "url": first_url})
            if len(out) >= max_results:
                break
        return out[:max_results]
    except Exception:
        return []


def _wikipedia_search(query: str, max_results: int = 5) -> List[dict]:
    """Wikipedia search API as a last-resort provider. Returns article URLs."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q}&format=json&utf8=1&srlimit={max_results}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json() if resp.status_code == 200 else {}
        results = data.get("query", {}).get("search", [])
        out: List[dict] = []
        for r in results:
            title = r.get("title")
            if not title:
                continue
            page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            out.append({"title": title, "url": page_url})
        return out[:max_results]
    except Exception:
        return []


def _google_news_search(query: str, max_results: int = 5) -> List[dict]:
    """Google News RSS (no key). Returns recent article titles and links."""
    try:
        q = urllib.parse.quote_plus(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        xml = resp.text or ""
        items = []
        for m in re.finditer(r"<item>\s*<title>([\s\S]*?)</title>[\s\S]*?<link>([\s\S]*?)</link>", xml, re.IGNORECASE):
            title = re.sub(r"<[^>]+>", "", m.group(1))
            link = re.sub(r"<[^>]+>", "", m.group(2))
            title = re.sub(r"\s+", " ", title).strip()
            link = link.strip()
            if title and link:
                items.append({"title": title, "url": link})
            if len(items) >= max_results:
                break
        return items
    except Exception:
        return []


def _fetch_readable(url: str, timeout: int = 15) -> str:
    """Fetch readable text using r.jina.ai proxy to avoid HTML parsing dependencies."""
    try:
        # Ensure scheme
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url
        proxy_url = "https://r.jina.ai/" + url
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(proxy_url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text or ""
    except Exception:
        pass
    # Fallback: basic HTML fetch and crude tag strip
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            html = resp.text or ""
            text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
    except Exception:
        pass
    return ""


def _answer_from_web(question: str, selected_model: str, gemini_key: str, groq_key: str, gemini_model_name: str, groq_model_name: str) -> dict | None:
    """Search the web and answer using only web sources. Returns a response dict or None."""
    results = _duckduckgo_search(question, max_results=5)
    if not results:
        # Prefer news when question appears newsy
        if any(w in question.lower() for w in ["news", "today", "latest", "happened", "breaking"]):
            results = _google_news_search(question, max_results=6)
    if not results:
        results = _bing_search(question, max_results=5)
    if not results:
        results = _ddg_instant(question, max_results=5)
    if not results:
        results = _wikipedia_search(question, max_results=5)
    if not results:
        return None
    # Deduplicate by URL and take top few
    seen = set()
    unique = []
    for r in results:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            unique.append(r)
        if len(unique) >= 4:
            break
    chosen = unique
    contents: List[str] = []
    used_urls: List[str] = []
    for r in chosen:
        text = _fetch_readable(r.get("url", ""))
        if text:
            # Trim to avoid overly long prompts
            trimmed = text.strip()
            if len(trimmed) > 4000:
                trimmed = trimmed[:4000]
            contents.append(f"Source: {r.get('title','')} ({r.get('url','')})\n\n{trimmed}")
            used_urls.append(r.get("url", ""))
        if len(contents) >= 3:
            break
    if not contents:
        return None
    web_context = "\n\n---\n\n".join(contents)
    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the web content below. "
        "Cite up to 2 sources by URL in parentheses. Be concise.\n\n"
        f"Question: {question}\n\n"
        f"Web content (from recent sources):\n{web_context}\n\n"
        "Answer:"
    )
    text = _call_llm(prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
    if not text:
        return None
    prefix = "From the web (sources: " + ", ".join(used_urls[:3]) + "): "
    return _add_to_history_and_return(prefix + text, "web", used_urls)


@api.post("/chat/ask")
def ask(payload: dict):
    question = payload.get("question", "").strip()
    if not question:
        return {"answer": "Please enter a question."}

    CHAT_HISTORY.append({"role": "user", "content": question})

    selected_model = payload.get("model", "gemini")
    web_only = bool(payload.get("web_only", False))
    gemini_key = settings.gemini_api_key
    groq_key = settings.groq_api_key
    gemini_model_name = settings.gemini_model or "gemini-2.5-flash"
    groq_model_name = settings.groq_model or "openai/gpt-oss-20b"

    # If user toggled web-only, bypass document context entirely
    if web_only:
        web_resp = _answer_from_web(question, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
        if web_resp:
            return web_resp
        return _add_to_history_and_return("I could not retrieve web results at the moment.", "web", [])

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


    retrieved = VECTOR_STORE.search(question, k=8)
    context_chunks = [c.text for _, c in retrieved]
    context_text = "\n\n".join(context_chunks[:6])
    sources = VECTOR_STORE.top_source_titles(question, k=6)

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
                # Fall back to web search if documents are insufficient
                web_resp = _answer_from_web(question, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
                if web_resp:
                    return web_resp
            prefix = "From your documents (sources: " + ", ".join(sources[:3]) + "): " if sources else "From your documents: "
            return _add_to_history_and_return(prefix + text, "docs", sources)

    # No useful context found at all; try web search as fallback
    web_resp = _answer_from_web(question, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
    if web_resp:
        return web_resp
    prefix = "I could not find an answer to your question in your documents, and web search failed. "
    prompt = (
        "Provide a concise, factual answer from general knowledge. Avoid fabrications and include a short explanation.\n\n"
        f"Question: {question}"
    )
    text = _call_llm(prompt, selected_model, gemini_key, groq_key, gemini_model_name, groq_model_name)
    if text:
        return _add_to_history_and_return(prefix + text, "general", [])
    
    model_name = "Gemini" if selected_model == "gemini" else ("Groq-openai" if selected_model == "groq-openai" else "Groq-qwen")
    return _add_to_history_and_return(f"I could not find an answer to your question in your documents. ({model_name} error; check API key/model and internet)", "general", [])
