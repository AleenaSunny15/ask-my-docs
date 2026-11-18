import pickle
import faiss
import numpy as np
from fastembed import TextEmbedding
from ingest import load_pdfs, chunk_pages

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_index():
    chunks = chunk_pages(load_pdfs())
    model = TextEmbedding(model_name=MODEL_NAME)

    texts = [c["text"] for c in chunks]
    vectors = np.array(list(model.embed(texts)), dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, "index.faiss")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {len(chunks)} chunks")

if __name__ == "__main__":
    build_index()