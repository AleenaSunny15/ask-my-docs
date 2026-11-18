import pickle
import faiss
import numpy as np
from fastembed import TextEmbedding

model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

question = input("Ask a question: ")
q_vec = np.array(list(model.embed([question])), dtype="float32")
faiss.normalize_L2(q_vec)

scores, ids = index.search(q_vec, 3)

for score, i in zip(scores[0], ids[0]):
    c = chunks[i]
    print(f"\n--- {c['source']} page {c['page']} (score {score:.2f}) ---")
    print(c["text"][:400])