import os
import time
import pickle
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
    
def retrieve(question, k=6):
    q_vec = np.array(list(embedder.embed([question])), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0]]


def answer(question, k=6):
    hits = retrieve(question, k)
    context = "\n\n".join(
        f"[{h['source']} p.{h['page']}] {h['text']}" for h in hits
    )
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
    headers = {"Authorization": f"Bearer {API_KEY}"}
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