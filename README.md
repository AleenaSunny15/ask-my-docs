# Ask My Docs: RAG Q&A over PDFs

Ask questions about a PDF and get answers with page citations.
Answers come only from the document; if it isn't there, the app says it doesn't know.

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
2. Put a text-based PDF in `data/`
3. Create a `.env` file containing `GROQ_API_KEY=your_key` (free key from console.groq.com)
4. `python src/embed.py` (builds the search index)
5. `streamlit run app.py`

## Tech
Python, FAISS, fastembed (ONNX), Groq API, Streamlit

## Author
Aleena Sunny - www.linkedin.com/in/aleena-sunny-450a63293 