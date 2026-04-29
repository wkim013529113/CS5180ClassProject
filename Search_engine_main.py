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

from Data_loader import load_all, stream_documents
from Inverted_indexer       import InvertedIndex, INDEX_FILENAME
from Retrieval_model_bm25        import BM25
from Evaluator   import evaluate
from Interactive_searcher_one_load      import run_search
import os

BOLD_SEP = "═" * 65
THIN_SEP = "─" * 65
CHUNK_SIZE = 1000   # docs per epoch during incremental build


# ── Boot ──────────────────────────────────────────────────────────────────────

def boot(data_dir: str) -> tuple:
    """
    Smart boot — load index from disk if it exists, build it if not.

    First run
    ---------
    1. Stream documents in chunks of CHUNK_SIZE (incremental build)
    2. Save the finished index to disk as index.pkl
    3. Load queries + qrels + all documents (for search display)

    Subsequent runs
    ---------------
    1. Detect index.pkl on disk
    2. Load it in ~0.5s instead of rebuilding (~5s)
    3. Load queries + qrels + all documents

    Returns (documents, queries, qrels, ranker)
    """
    print(BOLD_SEP)
    print("  Financial Document Search Engine")
    print("  CS 5180 — Information Retrieval System")
    print(BOLD_SEP)

    doc_path   = os.path.join(data_dir, "documents.json")
    index_path = os.path.join(data_dir, INDEX_FILENAME)

    # ── Index: load from disk or build fresh ─────────────────────────────────
    if InvertedIndex.exists(index_path):
        print(f"\n  [Boot] Index found on disk — loading...\n")
        idx = InvertedIndex.load(index_path)
    else:
        print(f"\n  [Boot] No index on disk — building incrementally "
              f"({CHUNK_SIZE} docs/epoch)...\n")
        idx = InvertedIndex()
        idx.build_incremental(doc_path, chunk_size=CHUNK_SIZE)
        idx.save(index_path)

    # ── Load queries, qrels, and full documents dict ──────────────────────────
    _, queries, qrels = load_all(data_dir)

    # documents dict needed by search.py for full-text display
    from Data_loader import load_documents
    documents = load_documents(doc_path)

    ranker = BM25(idx, k1=1.5, b=0.75)
    print(f"\n  Ready — {idx.N:,} documents indexed.\n")
    return documents, queries, qrels, ranker


# ── Menu ──────────────────────────────────────────────────────────────────────

def print_menu() -> None:
    print(BOLD_SEP)
    print("  What would you like to do?")
    print()
    print("    [1]  Run static evaluation   (AP per query + MAP)")
    print("    [2]  Interactive search      (type any query)")
    print("    [3]  Rebuild index           (delete disk index, rebuild fresh)")
    print("    [4]  Exit")
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


def handle_rebuild(data_dir: str) -> "InvertedIndex":
    """Mode 3 — delete disk index and rebuild from scratch."""
    index_path = os.path.join(data_dir, INDEX_FILENAME)
    if InvertedIndex.exists(index_path):
        os.remove(index_path)
        print(f"\n  [Rebuild] Deleted old index: {index_path}")
    print(f"\n  [Rebuild] Building fresh index ({CHUNK_SIZE} docs/epoch)...\n")
    doc_path = os.path.join(data_dir, "documents.json")
    idx = InvertedIndex()
    idx.build_incremental(doc_path, chunk_size=CHUNK_SIZE)
    idx.save(index_path)
    print(f"\n  [Rebuild] Done — new index saved to disk.\n")
    return idx


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # One-time smart boot
    documents, queries, qrels, ranker = boot(data_dir)

    # Menu loop
    while True:
        print_menu()
        choice = input("  Enter choice [1 / 2 / 3 / 4]: ").strip()

        if choice == "1":
            handle_evaluate(ranker, queries, qrels)

        elif choice == "2":
            handle_search(ranker, documents, queries, qrels)

        elif choice == "3":
            idx = handle_rebuild(data_dir)
            ranker = BM25(idx, k1=1.5, b=0.75)

        elif choice == "4":
            print("\n  Goodbye!\n")
            break

        else:
            print("\n  Invalid choice — please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()