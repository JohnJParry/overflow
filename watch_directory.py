"""Watch a directory for new PDF files and add them to the vector store.

Usage:
    python watch_directory.py --watch-dir ./pdfs --persist-dir ./chroma_store [--year-pattern "(\d{4})"]
"""

import os
import time
import argparse
import logging
from pathlib import Path
import hashlib
from typing import Set, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import from your existing code
from ingest import load_pdfs, split_docs, embed_and_store


class PDFHandler(FileSystemEventHandler):
    def __init__(self, watch_dir: Path, persist_dir: Path, year_pattern: str = None):
        self.watch_dir = watch_dir
        self.persist_dir = persist_dir
        self.year_pattern = year_pattern
        self.processed_files: Dict[str, str] = {}
        self.load_processed_files()
        
    def load_processed_files(self):
        """Load record of already processed files."""
        record_path = self.persist_dir / "processed_files.txt"
        if record_path.exists():
            with open(record_path, "r") as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(":")
                        if len(parts) == 2:
                            self.processed_files[parts[0]] = parts[1]
    
    def save_processed_files(self):
        """Save record of processed files."""
        record_path = self.persist_dir / "processed_files.txt"
        with open(record_path, "w") as f:
            for filename, file_hash in self.processed_files.items():
                f.write(f"{filename}:{file_hash}\n")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate a hash of file contents to detect changes."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_pdf(Path(event.src_path))
            
    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.process_pdf(Path(event.src_path))
    
    def process_pdf(self, pdf_path: Path):
        """Process a single PDF file if it's new or changed."""
        relative_path = pdf_path.name
        current_hash = self.get_file_hash(pdf_path)
        
        if relative_path in self.processed_files and self.processed_files[relative_path] == current_hash:
            logging.info(f"File {relative_path} already processed and unchanged.")
            return
            
        logging.info(f"Processing new or changed PDF: {relative_path}")
        
        # Load the single PDF
        docs = load_pdfs(pdf_path.parent, self.year_pattern, None, [pdf_path.name])
        
        if not docs:
            logging.warning(f"No content extracted from {relative_path}")
            return
            
        # Split and embed
        chunks = split_docs(docs)
        embed_and_store(chunks, self.persist_dir)
        
        # Record this file as processed
        self.processed_files[relative_path] = current_hash
        self.save_processed_files()
        logging.info(f"Successfully added {relative_path} to the vector store")
    
    def process_existing_files(self):
        """Process all existing PDF files in the watched directory."""
        for pdf_path in self.watch_dir.glob("*.pdf"):
            self.process_pdf(pdf_path)


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    
    parser = argparse.ArgumentParser(description="Watch directory for new PDFs and add to vector store")
    parser.add_argument(
        "--watch-dir", type=Path, required=True,
        help="Directory to watch for new PDF files"
    )
    parser.add_argument(
        "--persist-dir", type=Path,
        default=Path("chroma_store"),
        help="Directory where Chroma vector store is saved",
    )
    parser.add_argument(
        "--year-pattern", type=str,
        default=r"(\d{4})",
        help="Regex with capture group for year in filename (e.g. '(\\d{4})')",
    )
    
    args = parser.parse_args()
    
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY is not set.\n"
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )
    
    # Create directories if they don't exist
    args.watch_dir.mkdir(parents=True, exist_ok=True)
    args.persist_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the event handler and observer
    event_handler = PDFHandler(args.watch_dir, args.persist_dir, args.year_pattern)
    
    # Process any existing files first
    logging.info(f"Processing existing PDFs in {args.watch_dir}")
    event_handler.process_existing_files()
    
    # Start watching for new files
    observer = Observer()
    observer.schedule(event_handler, str(args.watch_dir), recursive=False)
    observer.start()
    
    logging.info(f"Started watching {args.watch_dir} for new PDF files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
