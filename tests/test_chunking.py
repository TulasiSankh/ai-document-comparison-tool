import pytest
from app.chunking import chunk_document

def test_chunking_normal_text():
    text = "Paragraph 1 is here and it is long enough.\n\nParagraph 2 is here and it is long enough."
    chunks = chunk_document(text, min_length=10)
    assert len(chunks) == 2
    assert "Paragraph 1" in chunks[0]
    assert "Paragraph 2" in chunks[1]

def test_chunking_empty_input():
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []

def test_chunking_small_text_merges():
    # 6 chars, 11 chars, 27 chars = 44 chars total
    text = "Short.\n\nAlso short.\n\nLonger paragraph goes here."
    chunks = chunk_document(text, min_length=20)
    # They should all merge into 1 chunk because the first two fall under min_length
    assert len(chunks) == 1
    assert "Short." in chunks[0]
    assert "Also short." in chunks[0]
    assert "Longer paragraph goes here." in chunks[0]
    
def test_chunking_trailing_small_text():
    text = "Longer paragraph goes here which is sufficient size.\n\nSmall"
    chunks = chunk_document(text, min_length=20)
    # The first one is >= 20, the second one is small. The code merges remainder onto the last chunk
    assert len(chunks) == 1
    assert "Small" in chunks[0]
