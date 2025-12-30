# Overflow RAG Toolkit

Tools for ingesting PDFs into a local Chroma vector store, querying them with
citations, and exploring trends over time. Includes a Streamlit chat UI,
CLI utilities, a directory watcher, and an evaluation script.

## Features
- Ingest PDFs into a persistent local vector store
- Query with citations (file name + page number)
- Streamlit chat UI for interactive Q&A
- Trend analysis across years based on filename patterns
- Watch a folder for new PDFs and auto-index them
- Evaluation harness for batch testing

## Requirements
- Python 3.9+
- An OpenAI API key exported as `OPENAI_API_KEY`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start
1) Put your PDFs in a folder, for example `./pdfs`
2) Ingest them into a Chroma store:
```bash
python ingest.py --pdf-dir ./pdfs --persist-dir ./chroma_store
```
3) Query the store:
```bash
python query.py "What are the main findings?"
```
4) Launch the chat UI:
```bash
streamlit run chat_app.py
```

## CLI Usage

### Ingest PDFs
```bash
python ingest.py --pdf-dir ./pdfs --persist-dir ./chroma_store
```

Optional flags:
- `--year-pattern "(\d{4})"` to extract a year from filenames
- `--test` to ingest only the first 3 PDFs

### Query with Optional Year Filters
```bash
python query.py "What are the main findings?"
python query.py --year 2021 "Revenue drivers in Q1?"
python query.py --min-year 2019 --max-year 2021 "Key trends over time?"
```

### Trend Analysis Across Years
```bash
python trend_analysis.py --pdf-dir ./pdfs --year-pattern "(\d{4})"
```

### Watch a Directory for New PDFs
```bash
python watch_directory.py --watch-dir ./pdfs --persist-dir ./chroma_store
```

### Evaluate the RAG System
```bash
python test_rag_system.py --test-questions test_questions.json --persist-dir ./chroma_store
```

The evaluation script writes a CSV for manual scoring and can generate
summary statistics and plots once scores are filled in.

## Environment
Set the API key before running:
```bash
export OPENAI_API_KEY=sk-...
```

## Notes
- The vector store path defaults to `./chroma_store` across tools.
- Citation format: `[filename.pdf-p3]` as shown in query and chat outputs.
