import io

class DocumentParseError(Exception):
    """Raised when a document cannot be parsed or yields no text."""
    pass

def _parse_pdf(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentParseError("pypdf is not installed.")

    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        text_blocks = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_blocks.append(page_text.strip())
        
        extracted_text = "\n\n".join(text_blocks)
        if not extracted_text.strip():
            raise DocumentParseError("PDF parsing yielded empty text. Please ensure the file is not corrupted or comprised solely of scanned images.")
        return extracted_text

    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse PDF: {e}") from e

def _parse_docx(raw_bytes: bytes) -> str:
    try:
        from docx import Document
    except ImportError:
        raise DocumentParseError("python-docx is not installed.")

    try:
        doc = Document(io.BytesIO(raw_bytes))
        text_blocks = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_blocks.append(para.text.strip())
                
        extracted_text = "\n\n".join(text_blocks)
        if not extracted_text.strip():
            raise DocumentParseError("DOCX parsing yielded empty text. Please ensure the file is valid.")
        return extracted_text

    except DocumentParseError:
        raise
    except Exception as e:
        raise DocumentParseError(f"Failed to parse DOCX: {e}") from e

def load_document(raw_bytes: bytes, filename: str) -> str:
    """
    Loads raw bytes and returns valid utf-8 string text.
    Handles size-verification on the output text buffer directly.
    """
    ext = filename.lower()
    
    if ext.endswith(".pdf"):
        text = _parse_pdf(raw_bytes)
    elif ext.endswith(".docx"):
        text = _parse_docx(raw_bytes)
    else:
        # Fallback to standard utf-8 decode
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raise DocumentParseError(f"Failed to decode TXT/MD file: {e}") from e

    # Apply global text constraints (Max 200 KB output)
    if len(text) > 200 * 1024:
        raise DocumentParseError("Extracted total text exceeds text processing bounds (200 KB).")
        
    return text
