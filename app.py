import streamlit as st
from pypdf import PdfReader
from src.ingest import chunk_pages
from src.rag import answer, build_index

MAX_PAGES = 60

st.set_page_config(page_title="Ask My Docs", page_icon="📄")
st.title("📄 Ask My Docs")
st.caption("Upload a text-based PDF, then ask questions. Answers come only "
           f"from the document, with page citations (first {MAX_PAGES} pages).")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to get started.")
else:
    if st.session_state.get("doc_name") != uploaded.name:
        reader = PdfReader(uploaded)
        pages = []
        for i, page in enumerate(reader.pages[:MAX_PAGES]):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": uploaded.name,
                              "page": i + 1})
        if not pages:
            st.error("No readable text found. Try a text-based PDF "
                     "(not a scan).")
            st.stop()
        with st.spinner("Reading and indexing the document..."):
            chunks = chunk_pages(pages)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = build_index(chunks)
            st.session_state["doc_name"] = uploaded.name

    question = st.text_input("Your question")
    if question:
        with st.spinner("Searching and thinking..."):
            ans, hits = answer(
                question,
                index=st.session_state["index"],
                chunks=st.session_state["chunks"],
            )
        st.markdown(ans)
        st.subheader("Sources")
        for h in hits:
            with st.expander(f"{h['source']} - page {h['page']}"):
                st.write(h["text"])