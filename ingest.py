"""Ingest PDF documents into a local Chroma vector store, with optional year metadata extraction.

Usage:
    python ingest.py --pdf-dir ./pdfs --persist-dir ./chroma_store [--year-pattern "(\d{4})"]

Environment:
    OPENAI_API_KEY must be set for OpenAI embedding model.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    # Static type checking imports for IDEs
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.schema import Document
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma


def load_pdfs(pdf_dir: Path, year_pattern: str = None, limit: int = None, specific_files: List[str] = None) -> List["Document"]:
    """Load PDFs in *pdf_dir* with langchain's PyPDFLoader.

    Optionally extract `year` metadata using a regex, and limit to the first
    `limit` PDF files (for quick tests), or process only specified files.

    Returns a list of langchain Documents with metadata including
    `source` (filename), `page`, and optional `year`.
    """
    import re
    from tqdm import tqdm
    # Try community loader first (new path), fallback to official
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        from langchain.document_loaders import PyPDFLoader

    pattern = re.compile(year_pattern) if year_pattern else None
    docs = []

    # Get all PDF paths or filter to specific ones
    if specific_files:
        pdf_paths = [pdf_dir / filename for filename in specific_files if (pdf_dir / filename).exists()]
    else:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        
    if limit is not None and limit > 0:
        pdf_paths = pdf_paths[:limit]

    from pypdf.errors import PdfStreamError

    for pdf_path in tqdm(pdf_paths, desc="Loading PDFs"):
        year = None
        if pattern:
            m = pattern.search(pdf_path.name)
            if m:
                try:
                    year = int(m.group(1))
                except ValueError:
                    year = None
        loader = PyPDFLoader(str(pdf_path))
        try:
            docs_iter = loader.load()
        except PdfStreamError as e:
            logging.warning(f"Failed to parse PDF '%s': %s", pdf_path.name, e)
            continue
        except Exception as e:
            logging.warning(f"Error loading PDF '%s': %s", pdf_path.name, e)
            continue

        for doc in docs_iter:
            doc.metadata["source"] = pdf_path.name
            if year is not None:
                doc.metadata["year"] = year
            docs.append(doc)
    return docs


def split_docs(docs: List["Document"]) -> List["Document"]:
    """Split raw docs into manageable chunks for embedding."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from tqdm import tqdm

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in tqdm(docs, desc="Splitting documents"):
        doc_chunks = splitter.split_documents([doc])
        chunks.extend(doc_chunks)
    return chunks


def embed_and_store(chunks: List["Document"], persist_dir: Path) -> None:
    """Embed *chunks* with OpenAIEmbeddings and persist in Chroma."""
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        from langchain.embeddings import OpenAIEmbeddings
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=str(persist_dir),
    )
    store.persist()


def main() -> None:
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(description="Ingest PDFs into vector store")
    parser.add_argument(
        "--pdf-dir", type=Path, required=True,
        help="Folder with PDF files"
    )
    parser.add_argument(
        "--persist-dir", type=Path,
        default=Path("chroma_store"),
        help="Directory where Chroma vector store will be saved",
    )
    parser.add_argument(
        "--year-pattern", type=str,
        default=r"(\d{4})",
        help="Regex with capture group for year in filename (e.g. '(\\d{4})')",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test run: ingest only the first 3 PDF files",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY is not set.\n"
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    limit = 3 if args.test else None
    logging.info("Loading PDFs from %s", args.pdf_dir.resolve())
    raw_docs = load_pdfs(args.pdf_dir, args.year_pattern, limit)
    logging.info("Loaded %d pages", len(raw_docs))

    logging.info("Splitting documents into chunks ...")
    chunks = split_docs(raw_docs)
    logging.info("Generated %d chunks", len(chunks))

    logging.info("Embedding and storing vectors in %s", args.persist_dir.resolve())
    embed_and_store(chunks, args.persist_dir)

    logging.info("Ingestion complete ✔")


if __name__ == "__main__":
    main()

