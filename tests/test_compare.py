import pytest
import numpy as np
from unittest.mock import patch
from app.compare import compare_documents, ChunkDiff

# A mock for the embedding that gives distinct, predictable vectors
def mock_get_embeddings_batch(chunks):
    # E.g. turn chunk string lengths into different unit vectors to simulate sim
    # For deterministic testing, let's just make sure identical strings yield identical vectors
    # and totally different strings yield orthogonal axes.
    vecs = []
    vector_dim = 10
    
    vocab = {}
    for text in chunks:
        if text not in vocab:
            vocab[text] = np.zeros(vector_dim)
            if len(vocab) - 1 < vector_dim:
                vocab[text][len(vocab) - 1] = 1.0 # unit vector
            else:
                vocab[text][0] = 0.5 # fallback dummy
        vecs.append(vocab[text])
    return np.array(vecs)
    
@patch("app.compare.get_embeddings_batch")
def test_compare_identical_documents(mock_embed):
    # Setup mock to just return dummy identical matrices if chunks match
    mock_embed.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Needs to be > 40 characters per paragraph to prevent merging
    text = "Paragraph one is here and it is very very very long enough.\n\nParagraph two is here and it is very very very long."
    # Chunks are the same
    diffs = compare_documents(text, text, explain=False)
    assert len(diffs) == 2
    assert diffs[0].status == "unchanged"
    assert diffs[1].status == "unchanged"
    assert diffs[0].similarity > 0.99
    
@patch("app.compare.get_embeddings_batch")
def test_compare_completely_different(mock_embed):
    def fake_embed(chunks):
        return np.array([[1.0, 0.0]]) if chunks[0] == "Apples" else np.array([[0.0, 1.0]])
    
    mock_embed.side_effect = fake_embed
    
    diffs = compare_documents("Apples", "Oranges", explain=False)
    # The vectors are orthogonal, score is 0.
    # Therefore, A is "removed" and B is "added" because similarity is < SIM_MODIFIED.
    assert len(diffs) == 2
    assert diffs[0].status == "removed"
    assert diffs[0].doc_a_text == "Apples"
    assert diffs[1].status == "added"
    assert diffs[1].doc_b_text == "Oranges"

@patch("app.compare.get_embeddings_batch")
def test_empty_documents(mock_embed):
    diffs = compare_documents("", "")
    assert len(diffs) == 0

@patch("app.compare.get_embeddings_batch")
def test_one_empty_document(mock_embed):
    diffs = compare_documents("Apples", "")
    assert len(diffs) == 1
    assert diffs[0].status == "removed"
    
    diffs2 = compare_documents("", "Apples")
    assert len(diffs2) == 1
    assert diffs2[0].status == "added"
