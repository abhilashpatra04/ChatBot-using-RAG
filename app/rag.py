from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import re

@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    title: str | None = None
    doc_type: str | None = None

class VectorStore:
    def __init__(self, persist_path: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2", collection_name: str = "default"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection_name = collection_name
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        try:
            
            self.collection = self.client.get_collection(name=self.collection_name, embedding_function=self.embedding_function)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.embedding_function)

    def reset(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name=self.collection_name, embedding_function=self.embedding_function)

    def delete_by_doc(self, doc_id: str):
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass

    def add_texts(self, doc_id: str, texts: List[str], title: str | None = None, doc_type: str | None = None):
        if not texts:
            return
        ids = [f"{doc_id}-{i}" for i in range(len(texts))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i, "title": title or "", "doc_type": doc_type or ""} for i in range(len(texts))]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def search(self, query: str, k: int = 5) -> List[Tuple[float, DocumentChunk]]:
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)
        out: List[Tuple[float, DocumentChunk]] = []
        for text, meta, dist in zip(docs, metas, distances):
            out.append((float(dist), DocumentChunk(doc_id=meta.get("doc_id", ""), text=text, title=meta.get("title") or None, doc_type=meta.get("doc_type") or None)))
        return out

    def diversified_search(self, query: str, top_n_per_doc: int = 3, max_docs: int = 4) -> Dict[str, List[DocumentChunk]]:
        candidates = self.search(query, k=top_n_per_doc * max_docs * 2)
        buckets: Dict[str, List[DocumentChunk]] = {}
        for _, chunk in candidates:
            buckets.setdefault(chunk.doc_id, [])
            if len(buckets[chunk.doc_id]) < top_n_per_doc:
                buckets[chunk.doc_id].append(chunk)
        doc_ids = list(buckets.keys())[:max_docs]
        return {doc_id: buckets[doc_id] for doc_id in doc_ids}

    def build_merged_context(self, query: str, top_n_per_doc: int = 3, max_docs: int = 4, separator: str = "\n\n") -> Tuple[str, List[str]]:
        grouped = self.diversified_search(query, top_n_per_doc=top_n_per_doc, max_docs=max_docs)
        titles: List[str] = []
        parts: List[str] = []
        for doc_id, chunks in grouped.items():
            title = chunks[0].title or doc_id if chunks else doc_id
            titles.append(title)
            for chunk in chunks:
                parts.append(chunk.text)
        return separator.join(parts), titles

    def top_source_titles(self, query: str, k: int = 5) -> List[str]:
        results = self.search(query, k=k)
        titles: List[str] = []
        seen = set()
        for _, chunk in results:
            t = chunk.title or chunk.doc_id
            if t not in seen:
                seen.add(t)
                titles.append(t)
            if len(titles) >= 3:
                break
        return titles


def split_into_chunks(text: str, max_chars: int = 1000, overlap_chars: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if chunks and overlap_chars > 0:
                overlap = chunks[-1][-overlap_chars:]
                current = (overlap + " " + sent).strip()
            else:
                current = sent
    if current:
        chunks.append(current)
    return chunks
