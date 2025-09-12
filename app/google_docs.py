from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from typing import List
from .auth import get_credentials

router = APIRouter(prefix="/docs", tags=["docs"])


def _drive(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _docs(creds: Credentials):
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def _sheets(creds: Credentials):
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _slides(creds: Credentials):
    return build("slides", "v1", credentials=creds, cache_discovery=False)


MIMES = {
    "docs": "application/vnd.google-apps.document",
    "sheets": "application/vnd.google-apps.spreadsheet",
    "slides": "application/vnd.google-apps.presentation",
    # 'drive' category represents Google Drive files we handle (PDFs via download+parse)
    "drive": "application/pdf",
}


@router.get("/list")
def list_docs(request: Request):
    creds = get_credentials(request)
    if not creds:
        return JSONResponse([], status_code=200)
    service = _drive(creds)
    files: List[dict] = []
    for ft, mime in MIMES.items():
        try:
            res = service.files().list(q=f"mimeType='{mime}' and trashed=false",
                                       pageSize=50,
                                       fields="files(id, name, modifiedTime, webViewLink, mimeType)",
                                       orderBy="modifiedTime desc").execute()
            for f in res.get("files", []):
                f["type"] = ft
                files.append(f)
        except Exception:
            continue
    files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
    return files


def extract_text_from_google_doc(creds: Credentials, doc_id: str) -> str:
    doc = _docs(creds).documents().get(documentId=doc_id).execute()
    content = []
    for elem in doc.get("body", {}).get("content", []):
        p = elem.get("paragraph")
        if not p:
            continue
        for e in p.get("elements", []):
            txt = e.get("textRun", {}).get("content")
            if txt:
                content.append(txt)
    return "".join(content)


def extract_text_from_sheet(creds: Credentials, sheet_id: str) -> str:
    svc = _sheets(creds)
    meta = svc.spreadsheets().get(spreadsheetId=sheet_id).execute()
    out = []
    for sht in meta.get("sheets", [])[:5]:
        title = sht.get("properties", {}).get("title", "Sheet")
        try:
            result = svc.spreadsheets().values().get(spreadsheetId=sheet_id, range=f"'{title}'!A1:Z1000").execute()
            values = result.get("values", [])
            if values:
                out.append(f"\n--- Sheet: {title} ---\n")
                for row in values:
                    out.append("\t".join(str(c) for c in row) + "\n")
        except Exception:
            continue
    return "".join(out)


def extract_text_from_slides(creds: Credentials, pres_id: str) -> str:
    svc = _slides(creds)
    pres = svc.presentations().get(presentationId=pres_id).execute()
    slides = pres.get("slides", [])
    out = []
    for i, slide in enumerate(slides):
        out.append(f"\n--- Slide {i+1} ---\n")
        for el in slide.get("pageElements", []):
            shp = el.get("shape")
            if shp and "text" in shp:
                for t in shp["text"].get("textElements", []):
                    tr = t.get("textRun")
                    if tr and tr.get("content"):
                        out.append(tr["content"])
    return "".join(out)


def extract_text_from_pdf(creds: Credentials, file_id: str) -> str:
    # Try simple export to text first (works for Google Docs, not for native PDFs)
    svc = _drive(creds)
    try:
        data = svc.files().export(fileId=file_id, mimeType="text/plain").execute()
        if isinstance(data, bytes):
            txt = data.decode("utf-8", errors="ignore")
            if txt.strip():
                return txt
        if isinstance(data, dict) and "body" in data:
            txt = str(data["body"]).strip()
            if txt:
                return txt
    except Exception:
        pass

    # Fallback: download the PDF and parse with pdfminer.six
    try:
        import io
        from googleapiclient.http import MediaIoBaseDownload
        from pdfminer.high_level import extract_text as pdf_extract_text

        buf = io.BytesIO()
        req = svc.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        buf.seek(0)
        text = pdf_extract_text(buf)
        return text or ""
    except Exception:
        return ""


def get_doc_title(creds: Credentials, doc_id: str) -> str:
    meta = _drive(creds).files().get(fileId=doc_id, fields="name").execute()
    return meta.get("name", doc_id)
