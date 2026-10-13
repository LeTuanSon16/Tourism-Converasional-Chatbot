"""Build the FAISS index from PDFs: `python -m chatbot.ingest [file.pdf ...]`."""
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

from chatbot import config
from chatbot.rag import get_embeddings


def read_pdf(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def build_index(pdf_paths: list[Path]) -> FAISS:
    splitter = CharacterTextSplitter(separator="\n", chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    docs = splitter.create_documents([read_pdf(p) for p in pdf_paths], [{"source": p.name} for p in pdf_paths])
    db = FAISS.from_documents(docs, get_embeddings())
    db.save_local(str(config.VECTOR_DB_PATH))
    return db


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or config.DEFAULT_PDFS
    db = build_index(paths)
    print(f"Indexed {db.index.ntotal} chunks from {len(paths)} file(s) -> {config.VECTOR_DB_PATH}")
