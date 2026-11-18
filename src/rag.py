import os
import time
import pickle
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
LLM_MODEL = "openai/gpt-oss-20b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedder = None
_saved = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectors = np.array(list(get_embedder().embed(texts)), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def load_saved():
    global _saved
    if _saved is None:
        index = faiss.read_index("index.faiss")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        _saved = (index, chunks)
    return _saved


def retrieve(question, index, chunks, k=6):
    q_vec = np.array(list(get_embedder().embed([question])), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0] if i != -1]


def answer(question, k=6, index=None, chunks=None):
    if index is None:
        index, chunks = load_saved()
    hits = retrieve(question, index, chunks, k)
    context = "\n\n".join(
        f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits
    )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "API error: GROQ_API_KEY is missing.", hits

    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": (
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, reply exactly: "
                "I don't know based on the provided documents. "
                "Otherwise write in plain text with no LaTeX, "
                "and end with the source file and page number."
            )},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(5):
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 429:
            break
        time.sleep(5 * (attempt + 1))

    if r.status_code != 200:
        return f"API error {r.status_code}: {r.text}", hits
    return r.json()["choices"][0]["message"]["content"], hits


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type quit): ")
        if q.strip().lower() == "quit":
            break
        ans, hits = answer(q)
        print("\nANSWER:\n" + ans)
        print("\nSOURCES:")
        for h in hits:
            print(f"- {h['source']} page {h['page']}")