import streamlit as st
from src.rag import answer

st.set_page_config(page_title="Ask My Docs", page_icon="📄")
st.title("📄 Ask My Docs")
st.caption("Ask questions about your PDFs. Answers come only from the "
           "documents, with page citations.")

question = st.text_input("Your question")

if question:
    with st.spinner("Searching and thinking..."):
        ans, hits = answer(question)
    st.markdown(ans)
    st.subheader("Sources")
    for h in hits:
        with st.expander(f"{h['source']} - page {h['page']}"):
            st.write(h["text"])
            