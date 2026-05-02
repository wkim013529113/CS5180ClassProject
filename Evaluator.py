"""
evaluator.py
------------
Evaluation metrics for the IR system.

Academic reference
------------------
Manning, C.D., Raghavan, P., & Schutze, H. (2008).
Introduction to Information Retrieval. Cambridge University Press.
Chapter 8: Evaluation in information retrieval.

Metrics implemented
-------------------
- Precision at rank k  (P@k)
- Average Precision    (AP)   per query
- Mean Average Precision (MAP) across all queries
"""

from __future__ import annotations
from Retrieval_model_bm25 import BM25                        # v2 — dedup + finance-aware
from Inverted_indexer import InvertedIndex              # v3 — manifest-based incremental
from Preprocessor import preprocess          # v2 — finance-aware normalization


# ── Core metric functions ─────────────────────────────────────────────────────

def precision_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """
    Fraction of relevant documents in the top-k results.

    P@k = |{relevant} ∩ {top-k}| / k

    Parameters
    ----------
    ranked_docs : list[str]
        Ordered list of doc_ids (rank 1 = index 0).
    relevant : set[str]
        Set of truly relevant doc_ids for this query.
    k : int
        Cutoff rank.

    Returns
    -------
    float  in [0.0, 1.0]
    """
    if k <= 0 or not ranked_docs:
        return 0.0
    top_k = ranked_docs[:k]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def average_precision(ranked_docs: list[str], relevant: set[str]) -> float:
    """
    Average Precision (AP) for a single query.

    AP = (1 / R) × Σ  P@k × rel(k)
                   k=1..N

    where:
        R      = total number of relevant documents (from qrels)
        rel(k) = 1 if document at rank k is relevant, else 0
        P@k    = precision computed at rank k

    Key property: AP rewards finding relevant documents EARLY.
    Finding all relevant docs at ranks 1,2,3 gives AP=1.0.
    Finding them at ranks 23,24,25 gives a much lower AP.

    Parameters
    ----------
    ranked_docs : list[str]
        Ordered list of doc_ids from ranker (rank 1 = index 0).
    relevant : set[str]
        Set of truly relevant doc_ids for this query (from qrels).

    Returns
    -------
    float  in [0.0, 1.0]
        0.0 if no relevant docs exist or none are retrieved.
    """
    R = len(relevant)
    if R == 0:
        return 0.0

    hits_so_far = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant:
            hits_so_far   += 1
            precision_sum += hits_so_far / rank   # P@rank only when relevant

    return precision_sum / R


def mean_average_precision(ap_scores: dict[str, float]) -> float:
    """
    Mean Average Precision (MAP) across all queries.

    MAP = (1 / |Q|) × Σ AP(q)

    Parameters
    ----------
    ap_scores : dict[str, float]
        { query_id → AP score }

    Returns
    -------
    float  in [0.0, 1.0]
    """
    if not ap_scores:
        return 0.0
    return sum(ap_scores.values()) / len(ap_scores)


# ── Full evaluation runner ────────────────────────────────────────────────────

def evaluate(
    ranker   : BM25,
    queries  : dict[str, str],
    qrels    : dict[str, set[str]],
    top_k    : int = 25,
    verbose  : bool = True,
) -> dict:
    """
    Run BM25 over all queries and compute AP + MAP.

    Parameters
    ----------
    ranker  : BM25
    queries : dict[str, str]    { query_id → query_text }
    qrels   : dict[str, set]    { query_id → set of relevant doc_ids }
    top_k   : int               number of results to retrieve per query
    verbose : bool              print per-query breakdown

    Returns
    -------
    dict with keys:
        'ap'  : dict[str, float]  — AP per query
        'map' : float             — MAP across all queries
    """
    ap_scores: dict[str, float] = {}

    if verbose:
        print("=" * 70)
        print(f"{'Query ID':<10} {'AP':>6}  {'Hits':>6}  Query text")
        print("=" * 70)

    for qid, query_text in sorted(queries.items()):
        results  = ranker.rank(query_text, top_k=top_k)
        ranked_docs = [doc_id for doc_id, _ in results]
        relevant    = qrels.get(qid, set())

        ap   = average_precision(ranked_docs, relevant)
        hits = sum(1 for doc_id in ranked_docs if doc_id in relevant)
        ap_scores[qid] = ap

        if verbose:
            bar   = "█" * int(ap * 20)
            short = query_text[:42] + "..." if len(query_text) > 45 else query_text
            print(f"  {qid:<10} {ap:>6.4f}  {hits:>3}/{len(relevant):<3}  {short}")

    map_score = mean_average_precision(ap_scores)

    if verbose:
        print("=" * 70)
        print(f"  {'MAP':<10} {map_score:>6.4f}")
        print("=" * 70)

    return {"ap": ap_scores, "map": map_score}


# ── User query mode: rank + display with pagination ───────────────────────────

def search_and_display(
    ranker    : BM25,
    documents : dict[str, str],
    query_text: str,
    top_k     : int = 25,
    page_size : int = 5,
    relevant  : set[str] | None = None,
) -> None:
    """
    Run a query and display ranked results with pagination.

    Shows 5 documents per page (per spec). Each result shows:
        - Rank and doc_id
        - BM25 score
        - Relevance label (if qrels provided)
        - Full document text

    Parameters
    ----------
    ranker     : BM25
    documents  : dict[str, str]    { doc_id → raw text }
    query_text : str               raw query string
    top_k      : int               total results to retrieve
    page_size  : int               results per page (default 5)
    relevant   : set[str] | None   if provided, marks relevant docs with ✓
    """
    results = ranker.rank(query_text, top_k=top_k)

    if not results:
        print(f"\n  No results found for: '{query_text}'")
        return

    total_pages = (len(results) + page_size - 1) // page_size
    query_tokens = preprocess(query_text)

    print(f"\n{'='*65}")
    print(f"  Query : {query_text}")
    print(f"  Tokens: {query_tokens}")
    print(f"  Found : {len(results)} results  |  {total_pages} pages")
    print(f"{'='*65}")

    page = 0
    while True:
        start = page * page_size
        end   = start + page_size
        page_results = results[start:end]

        print(f"\n  Page {page+1} of {total_pages}")
        print(f"  {'─'*61}")

        for rank, (doc_id, score) in enumerate(page_results, start=start+1):
            rel_marker = ""
            if relevant is not None:
                rel_marker = "  ✓ RELEVANT" if doc_id in relevant else "  ✗"

            text    = documents.get(doc_id, "[text not found]")
            preview = text[:300].replace("\n", " ")
            if len(text) > 300:
                preview += "..."

            print(f"\n  [{rank}] doc_id={doc_id}   score={score:.4f}{rel_marker}")
            print(f"  {preview}")
            print(f"  {'─'*61}")

        # Pagination controls
        print(f"\n  Page {page+1}/{total_pages} — ", end="")
        options = []
        if page > 0:
            options.append("[p] previous")
        if page < total_pages - 1:
            options.append("[n] next")
        options.append("[q] quit")
        print("  |  ".join(options))

        choice = input("  > ").strip().lower()
        if choice == "n" and page < total_pages - 1:
            page += 1
        elif choice == "p" and page > 0:
            page -= 1
        elif choice == "q":
            break
        else:
            print("  (invalid input, staying on current page)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from Data_loader import load_all

    data_dir      = sys.argv[1] if len(sys.argv) > 1 else "."
    index_path    = os.path.join(data_dir, "index.pkl")
    manifest_path = os.path.join(data_dir, "index_manifest.json")
    doc_path      = os.path.join(data_dir, "documents.json")

    documents, queries, qrels = load_all(data_dir)

    # Build or load index using v3 smart builder
    idx = InvertedIndex()
    if InvertedIndex.exists(index_path):
        print("[Evaluator] Loading existing index from disk...")
        idx = InvertedIndex.load(index_path)
    else:
        print("[Evaluator] No index found — building from scratch...")
        idx.build_smart(
            filepath      = doc_path,
            index_path    = index_path,
            manifest_path = manifest_path,
            chunk_size    = 1000,
        )

    ranker = BM25(idx, k1=1.5, b=0.75)

    # ── Mode 1: full static evaluation (MAP) ──────────────────────────────────
    print("\n[Mode 1] Static evaluation over all 25 labeled queries\n")
    results = evaluate(ranker, queries, qrels, top_k=25, verbose=True)
    print(f"\n  Final MAP = {results['map']:.4f}")

    # ── Mode 2: interactive user query ────────────────────────────────────────
    print("\n[Mode 2] Interactive search")
    print("  Type a query to search, or press Enter to skip.\n")
    user_query = input("  Enter query: ").strip()
    if user_query:
        search_and_display(
            ranker    = ranker,
            documents = documents,
            query_text= user_query,
            top_k     = 25,
            page_size = 5,
            relevant  = None,   # no labels for user queries
        )