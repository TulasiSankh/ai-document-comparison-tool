# AI Document Comparison Tool

Semantic diff for two documents — finds what changed in *meaning*, not just
in text, using local embeddings (Ollama) instead of a cloud API.

## How it works

1. **Chunk** each document into paragraph-level sections (`app/chunking.py`)
2. **Embed** every chunk locally with `nomic-embed-text` via Ollama (`app/embeddings.py`)
3. **Align** chunks across the two documents by cosine similarity — greedy
   best-match, in document-A order (`app/compare.py`)
4. **Classify** each aligned pair as `unchanged`, `modified`, `added`, or `removed`
   based on similarity thresholds
5. For `modified` pairs, ask a local chat model (`llama3.1`) to explain the
   change in plain English (`app/llm.py`)

## Setup

```bash
# 1. Install Ollama (if you haven't): https://ollama.com
ollama pull nomic-embed-text
ollama pull llama3.1

# 2. Python deps
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

**Streamlit UI** (fastest way to try it):
```bash
streamlit run streamlit_app.py
```

**FastAPI backend** (if you want an API instead / as well):
```bash
uvicorn app.main:app --reload
# then POST two files to http://localhost:8000/compare
```

## Tuning

- `app/compare.py`: `SIM_UNCHANGED` / `SIM_MODIFIED` thresholds — loosen or
  tighten depending on how sensitive you want the diff to be.
- `app/chunking.py`: currently paragraph-level. For finer-grained diffs, swap
  in sentence-level splitting (e.g. with a lightweight sentence tokenizer).
- `app/llm.py`: swap `CHAT_MODEL` for any model you've pulled locally.

## Where to take it next

- Swap the greedy aligner for a proper sequence-alignment algorithm
  (Needleman-Wunsch style) if documents get heavily reordered
- Add PDF/DOCX ingestion (extract text before chunking)
- Cache embeddings so re-comparing the same document doesn't re-embed
- Add a "confidence" flag when similarity sits right at a threshold boundary
