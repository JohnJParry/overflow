"""Generate year-by-year summaries and an overall trend report.

Assumes you have already ingested PDFs (so we can reuse the Document
objects). You *can* run this without a vector store if you point it
directly at the PDF folder.

Usage:
    python trend_analysis.py --pdf-dir ./pdfs --year-pattern "(\\d{4})"

The `--year-pattern` regex is applied to each filename; the first
capturing group is treated as the year (e.g. `2020_report.pdf`).
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


def load_pdfs(pdf_dir: Path):
    # Try community loader first (new path), fallback to official to preserve compatibility
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        from langchain.document_loaders import PyPDFLoader

    from tqdm import tqdm
    docs = []
    for pdf in tqdm(sorted(pdf_dir.glob("*.pdf")), desc="Loading PDFs"):
        loader = PyPDFLoader(str(pdf))
        for doc in loader.load():
            doc.metadata["source"] = pdf.name
            docs.append(doc)
    return docs


def group_by_year(docs, year_pattern: str):
    pattern = re.compile(year_pattern)
    from tqdm import tqdm
    years = defaultdict(list)
    for doc in tqdm(docs, desc="Grouping by year"):
        m = pattern.search(doc.metadata["source"])
        if m:
            try:
                year = int(m.group(1))
            except ValueError:
                continue
            years[year].append(doc.page_content)
    return years


def summarise_chunks(chunks, llm, word_limit=300):
    joined = "\n\n".join(chunks[:150])  # hard cap tokens
    prompt = (
        f"Give a concise {word_limit}-word summary of the main points "
        f"from the following text:\n{joined}"
    )
    return llm.predict(prompt)


def main():
    parser = argparse.ArgumentParser(description="Trend analysis across years")
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument(
        "--year-pattern",
        type=str,
        default=r"(\\d{4})",
        help="Regex with capture group for year in filename",
    )
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set")

    print("Loading PDF pages ...")
    docs = load_pdfs(args.pdf_dir)
    print(f"Loaded {len(docs)} pages")

    print("Grouping by year ...")
    groups = group_by_year(docs, args.year_pattern)
    print("Found years:", sorted(groups))

    # Prefer community import for chat model to avoid deprecation
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0)

    yearly_summaries = {}
    for year, texts in sorted(groups.items()):
        print(f"Summarising {year} with {len(texts)} pages ...")
        yearly_summaries[year] = summarise_chunks(texts, llm)

    print("Generating trend report ...")
    combined = "\n\n".join(f"{yr}: {s}" for yr, s in sorted(yearly_summaries.items()))
    trend_prompt = (
        "Based on the following yearly summaries, discuss the most "
        "important trends and how they evolved over time:\n" + combined
    )
    trend_report = llm.predict(trend_prompt)

    print("\n=== Yearly Summaries ===")
    for yr, summ in sorted(yearly_summaries.items()):
        print(f"\n{yr}\n{'-'*20}\n{summ}\n")

    print("\n=== Trend Report ===\n")
    print(trend_report)


if __name__ == "__main__":
    main()
