"""
index.py
--------
Builds an inverted index over the document collection.

Data structures produced
------------------------
index       : dict[str, dict[str, int]]
                term  →  { doc_id → term_frequency }

doc_lengths : dict[str, int]
                doc_id → number of tokens after preprocessing

avgdl       : float
                average document length across all docs (for BM25)

N           : int
                total number of documents (for BM25 IDF)

Knowledge used
--------------
- Inverted index theory (Manning et al., "Introduction to Information
  Retrieval", Ch. 1)
- Term Frequency (TF) — raw count of term occurrences per document
- Document Frequency (DF) — derivable from len(index[term])
- Document length statistics — required by BM25 length normalization
"""

import math
import time
from collections import defaultdict

from Preprocessor import preprocess


# ── Index class ───────────────────────────────────────────────────────────────

class InvertedIndex:
    """
    Inverted index with BM25-ready statistics.

    Attributes
    ----------
    index : dict[str, dict[str, int]]
        Core structure: term → { doc_id → term_frequency }

    doc_lengths : dict[str, int]
        doc_id → token count after preprocessing

    avgdl : float
        Mean document length across the entire collection.

    N : int
        Total number of documents indexed.
    """

    def __init__(self):
        # term → { doc_id → tf }
        self.index: dict[str, dict[str, int]] = defaultdict(dict)

        # doc_id → token count
        self.doc_lengths: dict[str, int] = {}

        self.avgdl: float = 0.0
        self.N: int = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, documents: dict[str, str]) -> None:
        """
        Preprocess every document and populate the index.

        Steps per document
        ------------------
        1. Preprocess text → list of stemmed tokens
        2. Record document length (token count)
        3. Count term frequencies via a local counter
        4. Write each (term, doc_id, tf) triple into the index

        Parameters
        ----------
        documents : dict[str, str]
            { doc_id → raw text }  from data_loader.load_documents()
        """
        print("[Index] Building inverted index ...")
        start = time.time()

        self.N = len(documents)
        total_tokens = 0

        for doc_id, text in documents.items():

            # Step 1 — preprocess
            tokens = preprocess(text)

            # Step 2 — document length
            dl = len(tokens)
            self.doc_lengths[doc_id] = dl
            total_tokens += dl

            # Step 3 — count term frequencies for this document
            tf_counter: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counter[token] += 1

            # Step 4 — write into the inverted index
            # index[term][doc_id] = frequency of term in doc_id
            for term, freq in tf_counter.items():
                self.index[term][doc_id] = freq

        # Step 5 — compute average document length
        self.avgdl = total_tokens / self.N if self.N > 0 else 0.0

        elapsed = time.time() - start
        self._print_stats(elapsed)

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> dict[str, int]:
        """
        Return the postings list for a term.

        Returns
        -------
        dict[str, int]
            { doc_id → tf } for all documents containing the term.
            Empty dict if term not in index.

        Example
        -------
        >>> idx.get_postings("mortgag")
        {'doc42': 3, 'doc17': 1, ...}
        """
        return self.index.get(term, {})

    def get_df(self, term: str) -> int:
        """
        Document frequency — how many documents contain this term.

        Used by BM25 to compute IDF:
            IDF = log( (N - df + 0.5) / (df + 0.5) + 1 )

        Higher df → term is common → lower IDF → lower contribution to score.
        """
        return len(self.index.get(term, {}))

    def get_idf(self, term: str) -> float:
        """
        BM25 IDF for a term (Robertson & Zaragoza, 2009).

        Formula
        -------
            IDF(t) = log( (N - df + 0.5) / (df + 0.5) + 1 )

        The +1 outside the log keeps IDF positive even when df > N/2
        (i.e. the term appears in more than half the collection).

        Returns 0.0 for unknown terms (not in index).
        """
        df = self.get_df(term)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _print_stats(self, elapsed: float) -> None:
        vocab_size   = len(self.index)
        total_postings = sum(len(v) for v in self.index.values())

        # Term coverage buckets
        df_counts = [len(v) for v in self.index.values()]
        rare_terms   = sum(1 for d in df_counts if d == 1)
        common_terms = sum(1 for d in df_counts if d > self.N * 0.1)

        print(f"[Index] Documents indexed : {self.N:,}")
        print(f"[Index] Vocabulary size   : {vocab_size:,} unique terms")
        print(f"[Index] Total postings    : {total_postings:,}")
        print(f"[Index] Avg doc length    : {self.avgdl:.1f} tokens")
        print(f"[Index] Rare terms (df=1) : {rare_terms:,}")
        print(f"[Index] Common terms(>10%): {common_terms:,}")
        print(f"[Index] Build time        : {elapsed:.2f}s\n")


# ── Quick sanity check when run directly ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from Data_loader_all_together import load_all

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    documents, queries, qrels = load_all(data_dir)

    idx = InvertedIndex()
    idx.build(documents)

    # ── Spot checks ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("SPOT CHECKS")
    print("=" * 55)

    # 1. Postings for a financial term
    term = "mortgag"   # stemmed form of "mortgage"
    postings = idx.get_postings(term)
    print(f"\nPostings for '{term}':")
    print(f"  df = {idx.get_df(term)} documents")
    print(f"  idf = {idx.get_idf(term):.4f}")
    sample_postings = dict(list(postings.items())[:5])
    print(f"  sample (doc_id → tf): {sample_postings}")

    # 2. IDF comparison — common vs rare term
    print(f"\nIDF comparison:")
    for t in ["invest", "mortgag", "xylophone", "401k", "loan"]:
        print(f"  '{t}': df={idx.get_df(t):4d}  idf={idx.get_idf(t):.4f}")

    # 3. Document length spot check
    sample_ids = list(documents.keys())[:3]
    print(f"\nDocument lengths (tokens after preprocessing):")
    for doc_id in sample_ids:
        print(f"  doc {doc_id}: {idx.doc_lengths[doc_id]} tokens")
    print(f"  avgdl: {idx.avgdl:.1f} tokens")