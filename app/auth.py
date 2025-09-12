from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from itsdangerous import URLSafeSerializer
from .config import settings
import os

# Allow HTTP redirect URIs for local development (oauthlib requirement)
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
# Relax scope mismatch errors when Google returns a superset of scopes
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

router = APIRouter(prefix="/auth", tags=["auth"])

SESSION_COOKIE = "rag_session"
serializer = URLSafeSerializer(settings.session_secret, salt="session")

def _build_flow():
    if not settings.google_client_id or not settings.google_client_secret or not settings.google_oauth_redirect:
        return None
    client_config = {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    scopes = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/documents.readonly",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/presentations.readonly",
    ]
    flow = Flow.from_client_config(client_config=client_config, scopes=scopes)
    flow.redirect_uri = settings.google_oauth_redirect
    return flow

@router.get("/login")
def login():
    flow = _build_flow()
    if flow is None:
        return JSONResponse({"error": "OAuth not configured", "auth_url": None})
    auth_url, state = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    token_state = serializer.dumps({"state": state})
    resp = JSONResponse({"auth_url": auth_url})
    resp.set_cookie(SESSION_COOKIE, token_state, httponly=True, samesite="lax")
    return resp

@router.get("/callback")
def callback(request: Request):
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return JSONResponse({"error": "Missing session"}, status_code=400)
    try:
        data = serializer.loads(cookie)
    except Exception:
        return JSONResponse({"error": "Bad session"}, status_code=400)

    flow = _build_flow()
    if flow is None:
        return JSONResponse({"error": "OAuth not configured"}, status_code=500)

    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    token_blob = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }
    session = serializer.dumps({"creds": token_blob})
    resp = RedirectResponse(url="/")
    resp.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="lax")
    return resp


def get_credentials(request: Request) -> Credentials | None:
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return None
    try:
        data = serializer.loads(cookie)
    except Exception:
        return None
    creds_blob = data.get("creds")
    if not creds_blob:
        return None
    creds = Credentials(**creds_blob)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
    return creds

@router.post("/logout")
def logout():
    resp = JSONResponse({"status": "logged out"})
    resp.delete_cookie(SESSION_COOKIE)
    return resp
