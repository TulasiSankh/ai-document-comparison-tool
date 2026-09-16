from dataclasses import asdict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .compare import compare_documents
from .document_loader import load_document, DocumentParseError

app = FastAPI(title="AI Document Comparison Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/compare")
async def compare_files(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    try:
        text_a = load_document(await file_a.read(), file_a.filename)
        text_b = load_document(await file_b.read(), file_b.filename)
    except DocumentParseError as e:
        return {"error": str(e)}
        
    diffs = compare_documents(text_a, text_b)
    return {"diffs": [asdict(d) for d in diffs]}


@app.post("/compare-text")
async def compare_text(payload: dict):
    text_a = payload.get("text_a", "")
    text_b = payload.get("text_b", "")
    diffs = compare_documents(text_a, text_b)
    return {"diffs": [asdict(d) for d in diffs]}


@app.get("/health")
async def health():
    return {"status": "ok"}
