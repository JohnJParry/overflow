"""CLI tool to query the vector store with natural language questions.

Supports optional filtering by year metadata (if ingested with --year-pattern).

Examples:
    # Simple query
    python query.py "What are the main findings of the 2020 report?"
    # Query documents from a specific year
    python query.py --year 2021 "Revenue drivers in Q1?"
    # Query documents between two years
    python query.py --min-year 2019 --max-year 2021 "Key trends over time?"
"""

import argparse
import os
from pathlib import Path


def build_qa_chain(store_dir: Path, k: int = 8, metadata_filter: dict = None):
    """Return a RetrievalQA chain configured with citations and optional metadata filtering."""

    # Prefer community imports for chat model, embeddings, and vectorstores to avoid deprecation
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
    except ImportError:
        from langchain.embeddings import OpenAIEmbeddings
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Reload the store
    store = Chroma(
        persist_directory=str(store_dir),
        embedding_function=OpenAIEmbeddings(),
    )

    prompt = PromptTemplate(
        template=(
            "You are a helpful analyst. Answer the question using ONLY the context "
            "provided, and cite each statement with the file name and page number "
            "in square brackets, e.g. [budget_2019.pdf-p3]. If the answer is not "
            "contained in the context, say 'I don't know.'\n"
            "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        input_variables=["question", "context"],
    )

    # Prepare retriever with metadata filter if provided
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    retriever = store.as_retriever(
        search_type="similarity", search_kwargs=search_kwargs
    )
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def main():
    parser = argparse.ArgumentParser(description="Query your PDF knowledge base")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument(
        "--store-dir", type=Path,
        default=Path("chroma_store"),
        help="Directory where the Chroma vector store is persisted",
    )
    parser.add_argument(
        "--top-k", type=int,
        default=8,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--year", type=int,
        help="Filter to documents from this exact year (requires year metadata)",
    )
    parser.add_argument(
        "--min-year", type=int,
        help="Filter to documents from this year or later",
    )
    parser.add_argument(
        "--max-year", type=int,
        help="Filter to documents this year or earlier",
    )

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Build optional metadata filter for year(s)
    metadata_filter = {}
    if args.year is not None:
        metadata_filter['year'] = args.year
    else:
        year_cond = {}
        if args.min_year is not None:
            year_cond['$gte'] = args.min_year
        if args.max_year is not None:
            year_cond['$lte'] = args.max_year
        if year_cond:
            metadata_filter['year'] = year_cond
    # Use None if no filters
    if not metadata_filter:
        metadata_filter = None
    chain = build_qa_chain(
        args.store_dir,
        k=args.top_k,
        metadata_filter=metadata_filter,
    )

    # Use invoke() to avoid deprecated __call__
    result = chain.invoke({"query": args.question})
    answer = result["result"]
    sources = {
        f"{d.metadata.get('source')}-p{d.metadata.get('page', 'NA')}": None
        for d in result["source_documents"]
    }

    print("\nAnswer:\n--------")
    print(answer)
    print("\nSources:")
    for src in sources.keys():
        print(" -", src)


if __name__ == "__main__":
    main()
