"""
RAG System Evaluation Script

This script runs test questions from a JSON file against your RAG system
and evaluates the performance, generating detailed test reports.

Usage:
    python test_rag_system.py --test-questions test_questions.json --persist-dir ./chroma_store [options]
"""

import argparse
import json
import time
import csv
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import your query functionality
from query import build_qa_chain


def load_test_questions(json_path):
    """Load test questions from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def filter_questions(questions_data, categories=None, complexity=None, ids=None):
    """Filter questions based on category, complexity, or specific IDs."""
    questions = questions_data['questions']
    
    if ids:
        return [q for q in questions if q['id'] in ids]
    
    filtered = questions
    
    if categories:
        filtered = [q for q in filtered if any(cat in q['category'] for cat in categories)]
    
    if complexity:
        filtered = [q for q in filtered if q['complexity'] in complexity]
        
    return filtered


def run_test(question, qa_chain, top_k=5, metadata_filter=None):
    """Run a single test question against the RAG system and measure performance."""
    start_time = time.time()
    
    # Execute the query
    result = qa_chain.invoke({"query": question['question']})
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Extract answer and sources
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    # Get sources in a clean format
    sources = [
        f"{d.metadata.get('source')}-p{d.metadata.get('page', 'NA')}"
        for d in source_docs
    ]
    
    return {
        "question_id": question['id'],
        "question": question['question'],
        "category": question['category'],
        "complexity": question['complexity'],
        "answer": answer,
        "sources": sources,
        "num_chunks": len(source_docs),
        "processing_time": processing_time,
        "word_count": len(answer.split())
    }


def save_test_results(results, output_path):
    """Save test results to a CSV file."""
    fieldnames = [
        'question_id', 'category', 'complexity', 'question', 
        'answer', 'sources', 'num_chunks', 'processing_time', 
        'word_count', 'accuracy_score', 'citation_score', 
        'completeness_score', 'notes'
    ]
    
    # Add missing fields for manual evaluation
    for result in results:
        result['accuracy_score'] = ""
        result['citation_score'] = ""
        result['completeness_score'] = ""
        result['notes'] = ""
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def generate_summary_report(results_file, questions_data, output_dir):
    """Generate a summary report with visualizations after manual scoring."""
    # Load the CSV with manually added scores
    df = pd.read_csv(results_file)
    
    # Check if scoring is complete
    if df['accuracy_score'].isnull().any():
        print("Warning: Not all questions have been scored for accuracy")
    
    # Convert scores to numeric
    score_columns = ['accuracy_score', 'citation_score', 'completeness_score']
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary statistics
    summary = {
        'total_questions': len(df),
        'avg_processing_time': df['processing_time'].mean(),
        'avg_num_chunks': df['num_chunks'].mean(),
        'avg_word_count': df['word_count'].mean(),
        'avg_accuracy': df['accuracy_score'].mean(),
        'avg_citation': df['citation_score'].mean(),
        'avg_completeness': df['completeness_score'].mean(),
        'by_complexity': df.groupby('complexity')[score_columns].mean().to_dict(),
        'by_category': df.groupby('category')[score_columns].mean().to_dict()
    }
    
    # Save summary statistics
    with open(f"{output_dir}/summary_stats.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate visualizations
    
    # 1. Scores by complexity
    plt.figure(figsize=(12, 8))
    scores_by_complexity = df.groupby('complexity')[score_columns].mean().reset_index()
    scores_by_complexity_melted = pd.melt(
        scores_by_complexity, 
        id_vars=['complexity'], 
        value_vars=score_columns,
        var_name='Metric', 
        value_name='Score'
    )
    
    complexities = ['low', 'medium', 'high', 'very high']
    scores_by_complexity_melted['complexity'] = pd.Categorical(
        scores_by_complexity_melted['complexity'], 
        categories=complexities, 
        ordered=True
    )
    scores_by_complexity_melted = scores_by_complexity_melted.sort_values('complexity')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='complexity', y='Score', hue='Metric', data=scores_by_complexity_melted)
    plt.title('Performance Metrics by Question Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Average Score (1-5)')
    plt.ylim(0, this_should_not_match_anythinig_.df_scores.values.max() + 0.5)
    plt.legend(title='Metric')
    plt.savefig(f"{output_dir}/scores_by_complexity.png")
    
    # 2. Processing time vs complexity
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='complexity', y='processing_time', data=df)
    plt.title('Processing Time by Question Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Processing Time (seconds)')
    plt.savefig(f"{output_dir}/processing_time_by_complexity.png")
    
    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ['processing_time', 'num_chunks', 'word_count'] + score_columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Metrics')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    
    # 4. Category performance
    plt.figure(figsize=(15, 10))
    scores_by_category = df.groupby('category')[score_columns].mean().reset_index()
    scores_by_category_melted = pd.melt(
        scores_by_category, 
        id_vars=['category'], 
        value_vars=score_columns,
        var_name='Metric', 
        value_name='Score'
    )
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x='category', y='Score', hue='Metric', data=scores_by_category_melted)
    plt.title('Performance Metrics by Question Category')
    plt.xlabel('Category')
    plt.ylabel('Average Score (1-5)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scores_by_category.png")
    
    print(f"Summary report generated in {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test RAG system with curated questions")
    parser.add_argument(
        "--test-questions", type=Path, required=True,
        help="Path to the JSON file with test questions"
    )
    parser.add_argument(
        "--persist-dir", type=Path, default=Path("chroma_store"),
        help="Directory where the Chroma vector store is persisted",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("test_results"),
        help="Directory to save test results",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--filter-category", type=str, nargs="+",
        help="Filter questions by category (e.g. 'Committee Papers' 'Basic Factual')",
    )
    parser.add_argument(
        "--filter-complexity", type=str, nargs="+", 
        choices=["low", "medium", "high", "very high"],
        help="Filter questions by complexity level",
    )
    parser.add_argument(
        "--question-ids", type=int, nargs="+",
        help="Run only specific question IDs",
    )
    parser.add_argument(
        "--generate-report", action="store_true",
        help="Generate a summary report (requires scored CSV from previous run)",
    )
    parser.add_argument(
        "--scored-results", type=Path,
        help="Path to the scored CSV results file (for report generation)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test questions
    questions_data = load_test_questions(args.test_questions)
    
    # Generate report from previous run if requested
    if args.generate_report:
        if not args.scored_results:
            print("Error: --scored-results is required when using --generate-report")
            return
        generate_summary_report(args.scored_results, questions_data, output_dir)
        return
    
    # Filter questions if requested
    filtered_questions = filter_questions(
        questions_data,
        categories=args.filter_category,
        complexity=args.filter_complexity,
        ids=args.question_ids
    )
    
    if not filtered_questions:
        print("No questions match the specified filters.")
        return
    
    print(f"Running {len(filtered_questions)} test questions...")
    
    # Initialize the QA chain
    qa_chain = build_qa_chain(args.persist_dir, k=args.top_k)
    
    # Run tests
    results = []
    for question in tqdm(filtered_questions):
        result = run_test(question, qa_chain, top_k=args.top_k)
        results.append(result)
        
        # Save intermediate results (in case of crash)
        if len(results) % 5 == 0:
            save_test_results(results, output_dir / "results_intermediate.csv")
    
    # Save final results
    results_file = output_dir / "results.csv"
    save_test_results(results, results_file)
    
    # Save test configuration
    config = {
        "timestamp": timestamp,
        "test_questions": str(args.test_questions),
        "persist_dir": str(args.persist_dir),
        "top_k": args.top_k,
        "filter_category": args.filter_category,
        "filter_complexity": args.filter_complexity,
        "question_ids": args.question_ids,
        "num_questions_tested": len(filtered_questions)
    }
    
    with open(output_dir / "test_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Testing complete. Results saved to: {results_file}")
    print("Please manually score the results for accuracy, citation correctness, and completeness.")
    print("After scoring, run this script with --generate-report --scored-results results.csv")


if __name__ == "__main__":
    main()
