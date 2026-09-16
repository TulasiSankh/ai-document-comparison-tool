from dataclasses import asdict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .compare import compare_documents

app = FastAPI(title="AI Document Comparison Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_upload(file: UploadFile) -> str:
    return file.file.read().decode("utf-8", errors="ignore")


@app.post("/compare")
async def compare_files(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    text_a = _read_upload(file_a)
    text_b = _read_upload(file_b)
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
