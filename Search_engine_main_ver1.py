"""
main.py
-------
Single entry point for the CS 5180 Financial IR System.

Boots once (load → index → ranker), then presents a menu:
    [1] Run static evaluation  → AP per query + MAP
    [2] Interactive search     → paginated results + new search loop
    [3] Exit

Usage
-----
    python main.py
    python main.py path/to/data/folder
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Data_loader_all_together import load_all
from Inverted_indexer_all_together       import InvertedIndex
from Retrieval_model_bm25_ver1        import BM25
from Evaluator_ver1   import evaluate
from Interactive_searcher_one_load      import run_search

BOLD_SEP = "═" * 65
THIN_SEP = "─" * 65


# ── Boot ──────────────────────────────────────────────────────────────────────

def boot(data_dir: str) -> tuple:
    """
    Load data and build index + ranker once.
    Returns (documents, queries, qrels, ranker).
    """
    print(BOLD_SEP)
    print("  Financial Document Search Engine")
    print("  CS 5180 — Information Retrieval System")
    print(BOLD_SEP)

    documents, queries, qrels = load_all(data_dir)

    idx = InvertedIndex()
    idx.build(documents)

    ranker = BM25(idx, k1=1.5, b=0.75)

    print(f"  Ready — {len(documents):,} documents indexed.\n")
    return documents, queries, qrels, ranker


# ── Menu ──────────────────────────────────────────────────────────────────────

def print_menu() -> None:
    print(BOLD_SEP)
    print("  What would you like to do?")
    print()
    print("    [1]  Run static evaluation   (AP per query + MAP)")
    print("    [2]  Interactive search      (type any query)")
    print("    [3]  Exit")
    print(BOLD_SEP)


def handle_evaluate(ranker, queries, qrels) -> None:
    """Mode 1 — run all 25 static queries and print AP + MAP."""
    print()
    results = evaluate(ranker, queries, qrels, top_k=25, verbose=True)
    print(f"\n  Final MAP = {results['map']:.4f}\n")


def handle_search(ranker, documents, queries, qrels) -> None:
    """Mode 2 — interactive search loop with pagination."""
    print()
    while True:
        print(THIN_SEP)
        raw = input("  Enter query (or 'back' to return to menu): ").strip()

        if not raw:
            print("  (empty query — please type something)\n")
            continue

        if raw.lower() in ("back", "b", "exit", "quit"):
            break

        run_search(
            query_text = raw,
            ranker     = ranker,
            documents  = documents,
            queries    = queries,
            qrels      = qrels,
        )

        print()
        again = input("  New search? [y / n]: ").strip().lower()
        if again != "y":
            break
        print()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # One-time boot
    documents, queries, qrels, ranker = boot(data_dir)

    # Menu loop
    while True:
        print_menu()
        choice = input("  Enter choice [1 / 2 / 3]: ").strip()

        if choice == "1":
            handle_evaluate(ranker, queries, qrels)

        elif choice == "2":
            handle_search(ranker, documents, queries, qrels)

        elif choice == "3":
            print("\n  Goodbye!\n")
            break

        else:
            print("\n  Invalid choice — please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()