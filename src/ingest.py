from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(folder="data"):
    pages = []
    for pdf in Path(folder).glob("*.pdf"):
        reader = PdfReader(pdf)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": pdf.name, "page": i + 1})
    return pages

def chunk_pages(pages, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = []
    for p in pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({"text": piece, "source": p["source"], "page": p["page"]})
    return chunks

if __name__ == "__main__":
    pages = load_pdfs()
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(chunks[0])