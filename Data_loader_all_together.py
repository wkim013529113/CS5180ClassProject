"""
data_loader.py
--------------
Loads the three dataset files for the IR system:
  - documents.json  → dict[doc_id, text]
  - queries.json    → dict[query_id, text]
  - qrels.json      → dict[query_id, set[doc_id]]
"""

import json
import os
from collections import defaultdict


def load_documents(filepath: str) -> dict[str, str]:
    """
    Load documents.json.

    Returns
    -------
    dict[str, str]
        { doc_id → text }  (5,000 entries)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = {}
    empty_count = 0

    for entry in raw:
        doc_id = str(entry["doc_id"])
        text   = entry.get("text", "").strip()

        if not text:
            empty_count += 1     # track but still store — ranker will score 0

        documents[doc_id] = text

    print(f"[Loader] Documents : {len(documents):,} loaded  |  {empty_count} empty texts")
    return documents


def load_queries(filepath: str) -> dict[str, str]:
    """
    Load queries.json.

    Returns
    -------
    dict[str, str]
        { query_id → query_text }  (25 entries)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    queries = {str(entry["query_id"]): entry["text"].strip() for entry in raw}

    print(f"[Loader] Queries   : {len(queries):,} loaded")
    return queries


def load_qrels(filepath: str) -> dict[str, set[str]]:
    """
    Load qrels.json.

    Returns
    -------
    dict[str, set[str]]
        { query_id → set of relevant doc_ids }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    qrels = defaultdict(set)
    for entry in raw:
        query_id = str(entry["query_id"])
        doc_id   = str(entry["doc_id"])
        # relevance is always 1 (binary), but guard just in case
        if int(entry.get("relevance", 1)) > 0:
            qrels[query_id].add(doc_id)

    total_judgments = sum(len(v) for v in qrels.values())
    print(f"[Loader] Qrels     : {len(qrels):,} queries  |  {total_judgments:,} relevant judgments total")
    return dict(qrels)


def load_all(data_dir: str = ".") -> tuple[dict, dict, dict]:
    """
    Convenience function — loads all three files at once.

    Parameters
    ----------
    data_dir : str
        Directory containing documents.json, queries.json, qrels.json.

    Returns
    -------
    (documents, queries, qrels)
    """
    doc_path   = os.path.join(data_dir, "documents.json")
    query_path = os.path.join(data_dir, "queries.json")
    qrels_path = os.path.join(data_dir, "qrels.json")

    for path in (doc_path, query_path, qrels_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Loader] Missing file: {path}")

    print(f"[Loader] Loading data from: {os.path.abspath(data_dir)}")
    documents = load_documents(doc_path)
    queries   = load_queries(query_path)
    qrels     = load_qrels(qrels_path)
    print("[Loader] All files loaded successfully.\n")

    return documents, queries, qrels


# ── quick sanity check when run directly ──────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    documents, queries, qrels = load_all(data_dir)

    # Spot-check
    sample_doc_id = next(iter(documents))
    print(f"Sample document  [{sample_doc_id}]: {documents[sample_doc_id][:120]}...")

    sample_qid = next(iter(queries))
    print(f"Sample query     [{sample_qid}]: {queries[sample_qid]}")

    sample_qrel_qid = next(iter(qrels))
    print(f"Sample qrel      [{sample_qrel_qid}]: {sorted(qrels[sample_qrel_qid])}")