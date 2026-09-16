import pytest
from unittest.mock import patch, MagicMock
from app.document_loader import load_document, DocumentParseError

def test_load_txt():
    # Valid text
    res = load_document(b"Hello world", "file.txt")
    assert res == "Hello world"
    
    # Valid md
    res = load_document(b"## Heading", "file.md")
    assert res == "## Heading"
    
def test_oversized_txt():
    big_bytes = b"a" * (200 * 1024 + 1)
    with pytest.raises(DocumentParseError) as e:
        load_document(big_bytes, "file.txt")
    assert "exceeds text processing bounds" in str(e.value)

@patch("pypdf.PdfReader")
def test_load_pdf_valid(mock_pdf_reader):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Text"
    mock_pdf_reader.return_value.pages = [mock_page]
    
    res = load_document(b"dummy_bytes", "file.pdf")
    assert res == "PDF Text"
    
@patch("docx.Document")
def test_load_docx_valid(mock_document):
    mock_para = MagicMock()
    mock_para.text = "DOCX Text"
    mock_document.return_value.paragraphs = [mock_para]
    
    res = load_document(b"dummy_bytes", "file.docx")
    assert res == "DOCX Text"

@patch("pypdf.PdfReader")
def test_empty_pdf(mock_pdf_reader):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "   "
    mock_pdf_reader.return_value.pages = [mock_page]
    
    with pytest.raises(DocumentParseError) as e:
        load_document(b"dummy_bytes", "file.pdf")
    assert "yielded empty text" in str(e.value)

@patch("docx.Document")
def test_empty_docx(mock_document):
    mock_para = MagicMock()
    mock_para.text = ""
    mock_document.return_value.paragraphs = [mock_para]
    
    with pytest.raises(DocumentParseError) as e:
        load_document(b"dummy_bytes", "file.docx")
    assert "yielded empty text" in str(e.value)
