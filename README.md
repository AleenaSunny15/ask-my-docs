# Ask My Docs: RAG Q&A over PDFs

Live demo: https://ask-my-docs-liomjrdbyvzzbhjne9bg9m.streamlit.app

Upload a text-based PDF and ask questions about it. You get answers with page
citations. Answers come only from the document; if the answer isn't there, the
app says it doesn't know.

## How it works
PDF -> chunks -> embeddings (fastembed, all-MiniLM-L6-v2) -> FAISS search
-> LLM answer (Groq API, free tier) -> Streamlit web app

## Evaluation
12 hand-written questions about the "Attention Is All You Need" paper
(including questions the paper cannot answer, to test refusals).

| Setting | Accuracy |
|---|---|
| chunk 800, k=4 | 83% |
| chunk 800, k=6 | 100% |

Notes: the question set was refined after the first run to remove unfair
keyword matches. With 12 questions, one question is about 8%, and settings
were tuned on the same questions, so results are indicative, not conclusive.

## Run it yourself
1. `pip install -r requirements.txt`
2. Create a `.env` file containing `GROQ_API_KEY=your_key` (free key from console.groq.com)
3. `streamlit run app.py`
4. Upload a text-based PDF on the web page and ask questions

To run the evaluation: put a PDF in `data/`, run `python src/embed.py`
(builds the search index), then `python evaluate.py`.

## Tech
Python, FAISS, fastembed (ONNX), Groq API, Streamlit

## Author
Aleena Sunny - https://github.com/AleenaSunny15