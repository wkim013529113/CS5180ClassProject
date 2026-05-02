"""
search.py
---------
Interactive search interface for the financial IR system.

Features
--------
- Loads data and builds index once at startup
- Accepts unlimited user queries in a loop
- Paginated results: 5 documents per page (n / p / q)
- After viewing results: prompts for a new search
- If the query matches a known static query (by text),
  relevant documents are marked with ✓

Usage
-----
    python search.py
    python search.py path/to/data/folder
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Data_loader  import load_all, load_documents
from Inverted_indexer_to_file        import InvertedIndex, INDEX_FILENAME
from Retrieval_model_bm25_ver1         import BM25
from Preprocessor_without_norm import preprocess


# ── Display constants ─────────────────────────────────────────────────────────
PAGE_SIZE  = 5
TOP_K      = 25
SEPARATOR  = "─" * 65
BOLD_SEP   = "═" * 65


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_relevant_for_query(
    query_text : str,
    queries    : dict[str, str],
    qrels      : dict[str, set[str]],
) -> tuple[str | None, set[str]]:
    """
    Check if this query exactly matches one of the 25 static labeled queries.
    Returns (query_id, relevant_set) or (None, empty set).
    """
    normalized = query_text.strip().lower()
    for qid, qtext in queries.items():
        if qtext.strip().lower() == normalized:
            return qid, qrels.get(qid, set())
    return None, set()


def display_page(
    page_results : list[tuple[str, float]],
    start_rank   : int,
    documents    : dict[str, str],
    relevant     : set[str],
) -> None:
    """Render one page of results to the terminal."""
    for rank, (doc_id, score) in enumerate(page_results, start=start_rank):
        rel_label = ""
        if relevant:
            rel_label = "  ✓ RELEVANT" if doc_id in relevant else "  ✗"

        # Full text with a 300-char preview
        text    = documents.get(doc_id, "[text not found]")
        preview = text[:300].replace("\n", " ").strip()
        if len(text) > 300:
            preview += "..."

        print(f"\n  [{rank:>2}]  doc_id = {doc_id:<10}  score = {score:.4f}{rel_label}")
        print(f"  {SEPARATOR}")
        print(f"  {preview}")
        print(f"  {SEPARATOR}")


def paginate(
    results   : list[tuple[str, float]],
    documents : dict[str, str],
    relevant  : set[str],
) -> None:
    """
    Drive the pagination loop for a result set.

    Controls
    --------
    n  → next page
    p  → previous page
    q  → quit back to search prompt
    """
    total      = len(results)
    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page        = 0

    while True:
        start = page * PAGE_SIZE
        end   = start + PAGE_SIZE
        page_results = results[start:end]

        print(f"\n  Page {page + 1} of {total_pages}  "
              f"(showing results {start + 1}–{min(end, total)} of {total})")

        display_page(
            page_results = page_results,
            start_rank   = start + 1,
            documents    = documents,
            relevant     = relevant,
        )

        # Build navigation prompt from what's available
        nav_options = []
        if page > 0:
            nav_options.append("[p] previous")
        if page < total_pages - 1:
            nav_options.append("[n] next")
        nav_options.append("[q] back to search")

        print(f"\n  {' | '.join(nav_options)}")
        choice = input("  > ").strip().lower()

        if choice == "n" and page < total_pages - 1:
            page += 1
        elif choice == "p" and page > 0:
            page -= 1
        elif choice == "q":
            break
        else:
            print("  (unrecognised input — staying on current page)")


def run_search(
    query_text : str,
    ranker     : BM25,
    documents  : dict[str, str],
    queries    : dict[str, str],
    qrels      : dict[str, set[str]],
) -> None:
    """
    Execute one full search: rank → header → paginate.
    """
    # Check if this is a known labeled query
    qid, relevant = find_relevant_for_query(query_text, queries, qrels)

    # Rank
    results      = ranker.rank(query_text, top_k=TOP_K)
    query_tokens = preprocess(query_text)

    # Header
    print(f"\n{BOLD_SEP}")
    print(f"  Query  : {query_text}")
    print(f"  Tokens : {query_tokens}")
    if qid:
        print(f"  Matched labeled query [{qid}]  "
              f"— {len(relevant)} relevant doc(s) known")
    else:
        print(f"  (unlabeled query — no relevance judgments available)")
    print(f"  Results: {len(results)} documents retrieved")
    print(f"{BOLD_SEP}")

    if not results:
        print("\n  No results found. Try different keywords.\n")
        return

    paginate(results, documents, relevant)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(data_dir: str = ".") -> None:
    """
    Entry point: smart boot, then loop on user queries.

    Boot logic (mirrors main.py)
    ----------------------------
    - index.pkl exists → load from disk (~0.06s, no rebuild)
    - index.pkl missing → build incrementally (1000 docs/epoch), then save
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
        print(f"\n  [Boot] No index on disk — building incrementally...\n")
        idx = InvertedIndex()
        idx.build_incremental(doc_path, chunk_size=1000)
        idx.save(index_path)

    # ── Load supporting data ──────────────────────────────────────────────────
    _, queries, qrels = load_all(data_dir)   # queries + qrels (tiny, always fast)
    documents = load_documents(doc_path)      # full text dict for display

    ranker = BM25(idx, k1=1.5, b=0.75)

    print(f"\n  System ready. {idx.N:,} documents indexed.")
    print(f"  Tip: type any of the 25 labeled queries exactly to see")
    print(f"       relevance markers (✓ / ✗) on results.\n")

    # ── Query loop ────────────────────────────────────────────────────────────
    while True:
        print(BOLD_SEP)
        raw = input("  Enter query (or 'exit' to quit): ").strip()

        if not raw:
            print("  (empty query — please type something)\n")
            continue

        if raw.lower() in ("exit", "quit", "q"):
            print("\n  Goodbye!\n")
            break

        run_search(
            query_text = raw,
            ranker     = ranker,
            documents  = documents,
            queries    = queries,
            qrels      = qrels,
        )

        # ── After viewing results: offer new search ───────────────────────────
        print()
        again = input("  New search? [y / n]: ").strip().lower()
        if again != "y":
            print("\n  Goodbye!\n")
            break
        print()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(data_dir)