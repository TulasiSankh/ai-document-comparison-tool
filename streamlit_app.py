import streamlit as st

from app.compare import compare_documents
from app.document_loader import load_document, DocumentParseError

st.set_page_config(page_title="AI Document Comparison Tool", layout="wide")
st.title("📄 AI Document Comparison Tool")
st.caption("Semantic diff powered by local embeddings (Ollama) — not just line-by-line text matching.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    file_a = st.file_uploader("Upload original", type=["txt", "md", "pdf", "docx"], key="upload_a")
    text_a = ""
    if file_a:
        try:
            if file_a.size > 5 * 1024 * 1024:
                st.error("File exceeds 5MB binary limit.")
                st.stop()
            text_a = load_document(file_a.read(), file_a.name)
        except getattr(DocumentParseError, "__module__", Exception) as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        text_a = st.text_area("...or paste original text", height=220, key="text_a")
        
with col2:
    st.subheader("Revised")
    file_b = st.file_uploader("Upload revised", type=["txt", "md", "pdf", "docx"], key="upload_b")
    text_b = ""
    if file_b:
        try:
            if file_b.size > 5 * 1024 * 1024:
                st.error("File exceeds 5MB binary limit.")
                st.stop()
            text_b = load_document(file_b.read(), file_b.name)
        except getattr(DocumentParseError, "__module__", Exception) as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        text_b = st.text_area("...or paste revised text", height=220, key="text_b")

STATUS_STYLE = {
    "unchanged": ("⚪", "#666666"),
    "modified": ("🟡", "#b8860b"),
    "added": ("🟢", "#1a7f37"),
    "removed": ("🔴", "#cf222e"),
}

if st.button("Compare documents", type="primary"):
    # Reject oversized files before processing (approx 200KB limit for demo)
    if len(text_a) > 200 * 1024 or len(text_b) > 200 * 1024:
        st.error("One of the documents is too large. Please limit files to 200KB.")
        st.stop()
        
    if not text_a.strip() or not text_b.strip():
        st.warning("Please provide valid text for both documents. Empty or purely whitespace documents are not allowed.")
    else:
        with st.spinner("Embedding chunks and computing semantic diff... (first run pulls models into memory, be patient)"):
            try:
                diffs = compare_documents(text_a, text_b)
            except RuntimeError as re:
                st.error(f"Error during parallel embedding: {re}")
                st.stop()
            except Exception as e:
                st.error(
                    f"Couldn't reach Ollama at localhost:11434 — is it running? "
                    f"(`ollama serve`, and `ollama pull nomic-embed-text` / `ollama pull llama3.1`)\n\n{e}"
                )
                st.stop()

        counts = {"unchanged": 0, "modified": 0, "added": 0, "removed": 0}
        for d in diffs:
            counts[d.status] += 1

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unchanged", counts["unchanged"])
        m2.metric("Modified", counts["modified"])
        m3.metric("Added", counts["added"])
        m4.metric("Removed", counts["removed"])

        st.divider()

        for d in diffs:
            icon, color = STATUS_STYLE[d.status]
            with st.container(border=True):
                header = f"**{icon} {d.status.upper()}**"
                if d.similarity is not None:
                    header += f"  ·  similarity: {d.similarity:.2f}"
                st.markdown(header)

                c1, c2 = st.columns(2)
                with c1:
                    if d.doc_a_text:
                        st.markdown(f"<div style='color:{color}'>{d.doc_a_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("*— not present —*")
                with c2:
                    if d.doc_b_text:
                        st.markdown(f"<div style='color:{color}'>{d.doc_b_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("*— not present —*")

                if d.explanation:
                    st.info(d.explanation)
