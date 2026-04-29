"""
index.py  (v2 -- incremental)
------------------------------
Inverted index with incremental (epoch-by-epoch) building.

What changed from v1
--------------------
v1: build(documents)
    Takes the full documents dict at once.
    Computes avgdl only at the end.

v2: add_batch(chunk)
    Accepts one chunk (e.g. 1,000 docs) at a time.
    Updates N, doc_lengths, and total_tokens incrementally.
    avgdl is recomputed after every batch -- always accurate.

    build_incremental(filepath, chunk_size)
    Drives the full streaming pipeline in one call.
    Calls stream_documents() -> add_batch() per epoch.

Why incremental avgdl matters
------------------------------
avgdl is used in BM25 length normalization.
If we only compute it at the very end (v1), we cannot use the index
for querying mid-build (e.g. for a live search engine that accepts
queries while still indexing new documents).

With v2, avgdl is always correct for the documents seen so far:
    avgdl = total_tokens_seen / N_seen

Attributes added
----------------
_total_tokens : int   running sum of all token counts (for avgdl)

Knowledge used
--------------
- Incremental index update (Buttcher et al., "Information Retrieval",
  Ch. 4 -- Dynamic Indexing)
- Online mean computation: new_avg = total / count  (no need to store
  all lengths, just the running sum)
"""

import math
import time
import pickle
import os
from collections import defaultdict

from Preprocessor import preprocess

# Default filename written next to the data files
INDEX_FILENAME = "index.pkl"


class InvertedIndex:
    """
    Inverted index with BM25-ready statistics.
    Supports both full batch build and incremental epoch-by-epoch build.

    Attributes
    ----------
    index        : dict[str, dict[str, int]]
                   term -> { doc_id -> term_frequency }
    doc_lengths  : dict[str, int]
                   doc_id -> token count after preprocessing
    avgdl        : float   mean document length (updated after every batch)
    N            : int     total documents indexed so far
    _total_tokens: int     running token sum (private, used for avgdl)
    """

    def __init__(self):
        self.index         : dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_lengths   : dict[str, int]            = {}
        self.avgdl         : float                     = 0.0
        self.N             : int                       = 0
        self._total_tokens : int                       = 0   # NEW in v2

    # ── Incremental batch update (NEW in v2) ──────────────────────────────────

    def add_batch(self, chunk: dict[str, str]) -> dict:
        """
        Index one batch (epoch) of documents.

        Called once per chunk by build_incremental().
        Can also be called manually for custom streaming pipelines.

        Steps
        -----
        1. For each doc in chunk: preprocess -> count TF -> write to index
        2. Update running totals: N, _total_tokens
        3. Recompute avgdl from running totals (always accurate)

        Parameters
        ----------
        chunk : dict[str, str]
            { doc_id -> raw text } -- one chunk from stream_documents()

        Returns
        -------
        dict  batch stats:
            docs_added  : int   docs added this batch
            tokens_added: int   tokens processed this batch
            vocab_size  : int   current vocabulary size after this batch
        """
        batch_tokens = 0

        for doc_id, text in chunk.items():

            # Skip if already indexed (handles duplicate doc_ids gracefully)
            if doc_id in self.doc_lengths:
                continue

            # Step 1 -- preprocess
            tokens = preprocess(text)
            dl     = len(tokens)

            # Step 2 -- document length
            self.doc_lengths[doc_id]  = dl
            self._total_tokens       += dl
            batch_tokens             += dl

            # Step 3 -- term frequencies for this doc
            tf_counter: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counter[token] += 1

            # Step 4 -- write to index
            for term, freq in tf_counter.items():
                self.index[term][doc_id] = freq

        # Step 5 -- update global counts and recompute avgdl
        self.N     += len(chunk)
        self.avgdl  = self._total_tokens / self.N if self.N > 0 else 0.0

        return {
            "docs_added"  : len(chunk),
            "tokens_added": batch_tokens,
            "vocab_size"  : len(self.index),
        }

    # ── Full incremental build driver (NEW in v2) ─────────────────────────────

    def build_incremental(
        self,
        filepath   : str,
        chunk_size : int = 1000,
    ) -> None:
        """
        Build the index by streaming documents in epochs of chunk_size.

        Replaces v1's build(documents) for large collections.
        Each epoch: load chunk -> add_batch -> free chunk from RAM.

        Parameters
        ----------
        filepath   : str   path to documents.json
        chunk_size : int   docs per epoch (default 1000)
        """
        from Data_loader import stream_documents

        print(f"[Index] Incremental build  |  chunk_size={chunk_size}")
        print(f"[Index] {'Epoch':<8} {'Docs':>6} {'Total':>7} "
              f"{'Vocab':>7} {'avgdl':>7} {'Time':>6}")
        print(f"[Index] {'-'*52}")

        start_all = time.time()
        epoch     = 0

        for chunk in stream_documents(filepath, chunk_size=chunk_size):
            epoch     += 1
            t0         = time.time()
            stats      = self.add_batch(chunk)
            elapsed    = time.time() - t0

            print(f"[Index] {epoch:<8} "
                  f"{stats['docs_added']:>6,} "
                  f"{self.N:>7,} "
                  f"{stats['vocab_size']:>7,} "
                  f"{self.avgdl:>7.1f} "
                  f"{elapsed:>5.2f}s")

        total_elapsed = time.time() - start_all
        print(f"[Index] {'-'*52}")
        self._print_stats(total_elapsed)

    # ── v1-compatible full build (kept for backward compatibility) ─────────────

    def build(self, documents: dict[str, str]) -> None:
        """
        v1 interface -- index a full documents dict at once.
        Internally calls add_batch() in one shot.
        Kept so that existing code using build() still works unchanged.
        """
        print("[Index] Building index (single batch) ...")
        start = time.time()
        self.add_batch(documents)
        self._print_stats(time.time() - start)

    # ── Disk persistence (NEW in v3) ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialize the entire index to disk using pickle.

        What gets saved
        ---------------
        - self.index         (term -> { doc_id -> tf })
        - self.doc_lengths   (doc_id -> token count)
        - self.avgdl         (float)
        - self.N             (int)
        - self._total_tokens (int)

        Why pickle
        ----------
        pickle serializes arbitrary Python objects directly — no schema
        needed, no data loss, and for a dict of dicts it is 3-5x faster
        and more compact than json. The tradeoff is that the file is
        binary and Python-only.

        Production alternative: Lucene's binary posting list format,
        which uses delta-encoded, compressed integers and supports
        memory-mapped random access. For our scale pickle is sufficient.

        Parameters
        ----------
        path : str
            Full file path to write (e.g. "data/index.pkl")
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        payload = {
            "index"        : dict(self.index),
            "doc_lengths"  : self.doc_lengths,
            "avgdl"        : self.avgdl,
            "N"            : self.N,
            "_total_tokens": self._total_tokens,
        }

        start = time.time()
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(path) / (1024 * 1024)
        elapsed = time.time() - start
        print(f"[Index] Saved  → {path}  "
              f"({size_mb:.1f} MB  |  {elapsed:.2f}s)")

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        """
        Deserialize an index from disk.

        Returns a fully populated InvertedIndex ready for querying —
        no build step needed.

        Parameters
        ----------
        path : str
            File path written by save() (e.g. "data/index.pkl")

        Returns
        -------
        InvertedIndex

        Raises
        ------
        FileNotFoundError if path does not exist.

        Example
        -------
        >>> idx = InvertedIndex.load("data/index.pkl")
        >>> ranker = BM25(idx)
        >>> results = ranker.rank("mortgage rates")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Index] No index file at: {path}")

        start = time.time()
        with open(path, "rb") as f:
            payload = pickle.load(f)

        idx                = cls()
        idx.index          = defaultdict(dict, payload["index"])
        idx.doc_lengths    = payload["doc_lengths"]
        idx.avgdl          = payload["avgdl"]
        idx.N              = payload["N"]
        idx._total_tokens  = payload["_total_tokens"]

        elapsed = time.time() - start
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Index] Loaded ← {path}  "
              f"({size_mb:.1f} MB  |  {idx.N:,} docs  |  {elapsed:.2f}s)")
        return idx

    @staticmethod
    def exists(path: str) -> bool:
        """Return True if a saved index file exists at path."""
        return os.path.exists(path)

    # ── Lookup helpers (unchanged from v1) ────────────────────────────────────

    def get_postings(self, term: str) -> dict[str, int]:
        """Return { doc_id -> tf } for all docs containing term."""
        return self.index.get(term, {})

    def get_df(self, term: str) -> int:
        """Document frequency -- how many docs contain this term."""
        return len(self.index.get(term, {}))

    def get_idf(self, term: str) -> float:
        """
        BM25 IDF:  log( (N - df + 0.5) / (df + 0.5) + 1 )
        Returns 0.0 for unknown terms.
        """
        df = self.get_df(term)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _print_stats(self, elapsed: float) -> None:
        vocab_size      = len(self.index)
        total_postings  = sum(len(v) for v in self.index.values())
        df_counts       = [len(v) for v in self.index.values()]
        rare_terms      = sum(1 for d in df_counts if d == 1)
        common_terms    = sum(1 for d in df_counts if d > self.N * 0.1)

        print(f"[Index] Documents indexed : {self.N:,}")
        print(f"[Index] Vocabulary size   : {vocab_size:,} unique terms")
        print(f"[Index] Total postings    : {total_postings:,}")
        print(f"[Index] Avg doc length    : {self.avgdl:.1f} tokens")
        print(f"[Index] Rare terms (df=1) : {rare_terms:,}")
        print(f"[Index] Common terms(>10%): {common_terms:,}")
        print(f"[Index] Build time        : {elapsed:.2f}s\n")


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else "."
    doc_path   = os.path.join(data_dir, "documents.json")
    index_path = os.path.join(data_dir, INDEX_FILENAME)

    print("=" * 60)
    print("DISK PERSISTENCE TEST")
    print("=" * 60 + "\n")

    # Step 1 -- build incrementally
    print("-- Step 1: Build incrementally (chunk_size=1000) --")
    idx = InvertedIndex()
    idx.build_incremental(doc_path, chunk_size=1000)

    # Step 2 -- save to disk
    print("-- Step 2: Save to disk --")
    idx.save(index_path)

    # Step 3 -- load from disk into a fresh object
    print("\n-- Step 3: Load from disk (simulates restart) --")
    idx2 = InvertedIndex.load(index_path)

    # Step 4 -- verify everything matches
    print("\n-- Step 4: Verify loaded index matches original --")
    checks = [
        ("N",      idx.N,      idx2.N),
        ("avgdl",  round(idx.avgdl, 4),  round(idx2.avgdl, 4)),
        ("vocab",  len(idx.index), len(idx2.index)),
        ("mortgag df", idx.get_df("mortgag"), idx2.get_df("mortgag")),
        ("401k idf",   round(idx.get_idf("401k"), 4),
                       round(idx2.get_idf("401k"), 4)),
    ]
    all_ok = True
    for label, original, loaded in checks:
        ok = original == loaded
        all_ok = all_ok and ok
        status = "✓" if ok else "✗ MISMATCH"
        print(f"  {label:<18} original={original}  loaded={loaded}  {status}")

    print(f"\n  All checks passed: {all_ok}")