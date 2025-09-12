from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from .routes import api, VECTOR_STORE

app = FastAPI(title="RAG Chatbot with Google Docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reset vector store on startup to avoid stale chunks after reloads
@app.on_event("startup")
def _reset_vectors():
    try:
        VECTOR_STORE.reset()
        print("ChromaDB collection reset on startup.")
    except Exception as e:
        print(f"Error resetting ChromaDB: {e}")

app.include_router(api)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
