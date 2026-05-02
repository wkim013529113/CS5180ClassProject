"""
index.py  (v3 — manifest-based incremental)
---------------------------------------------
Inverted index with manifest-protected incremental building.

What changed from v2
--------------------
v2: add_batch() skips duplicate doc_ids using doc_lengths check.
    BUT: still streams ALL documents every run to find new ones.
    No record of what's already indexed outside the pkl file.

v3: Adds a manifest file (index_manifest.json) alongside index.pkl.
    The manifest stores the set of all indexed doc_ids.
    On each run:
      - If no pkl exists        → full build → save pkl + manifest
      - If pkl exists, no changes → skip entirely (0 work)
      - If new docs detected    → load pkl → add new only → save both
      - If --rebuild flag       → delete pkl + manifest → full rebuild

    This means adding 1,000 new docs to a 5,000-doc index costs
    exactly 1,000 doc preprocessing operations — not 6,000.

Why a separate manifest file
-----------------------------
The manifest is a plain JSON file storing only doc_ids (not the full
index). This allows fast O(1) set-difference lookups to identify new
documents WITHOUT loading the full binary pkl into memory first.

Attributes added
----------------
MANIFEST_FILENAME : str   companion file to INDEX_FILENAME
"""

import math
import time
import pickle
import os
import json
from collections import defaultdict

from Preprocessor import preprocess          # v2 finance-aware preprocessor

INDEX_FILENAME    = "index.pkl"
MANIFEST_FILENAME = "index_manifest.json"    # NEW in v3


class InvertedIndex:
    """
    Inverted index with BM25-ready statistics.

    Supports:
      - Full incremental build (stream 1,000 docs at a time)
      - Manifest-based smart update (only index NEW documents)
      - Disk persistence with pkl + manifest companion file
      - Force rebuild option

    Attributes
    ----------
    index        : dict[str, dict[str, int]]
                   term -> { doc_id -> term_frequency }
    doc_lengths  : dict[str, int]
                   doc_id -> token count after preprocessing
    avgdl        : float   mean document length (updated after every batch)
    N            : int     total documents indexed so far
    _total_tokens: int     running token sum (used for avgdl)
    """

    def __init__(self):
        self.index          : dict[str, dict[str, int]] = defaultdict(dict)
        self.doc_lengths    : dict[str, int]            = {}
        self.avgdl          : float                     = 0.0
        self.N              : int                       = 0
        self._total_tokens  : int                       = 0

    # ── Manifest helpers (NEW in v3) ──────────────────────────────────────────

    @staticmethod
    def load_manifest(manifest_path: str) -> set[str]:
        """
        Load the set of already-indexed doc_ids from manifest file.

        Returns empty set if manifest does not exist.

        Parameters
        ----------
        manifest_path : str   path to index_manifest.json

        Returns
        -------
        set[str]   doc_ids already in the index
        """
        if not os.path.exists(manifest_path):
            return set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("indexed_ids", []))

    @staticmethod
    def save_manifest(manifest_path: str, doc_ids: set[str]) -> None:
        """
        Save the current set of indexed doc_ids to manifest file.

        Parameters
        ----------
        manifest_path : str     path to write
        doc_ids       : set[str] all doc_ids currently in the index
        """
        os.makedirs(
            os.path.dirname(manifest_path) if os.path.dirname(manifest_path) else ".",
            exist_ok=True
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"indexed_ids": sorted(doc_ids)}, f)

    # ── Incremental batch update ───────────────────────────────────────────────

    def add_batch(self, chunk: dict[str, str]) -> dict:
        """
        Index one batch of documents, skipping already-indexed doc_ids.

        Steps
        -----
        1. Skip docs already in self.doc_lengths (duplicate protection)
        2. Preprocess each new doc → count TF → write to index
        3. Update running totals: N, _total_tokens
        4. Recompute avgdl from running totals (always accurate)

        Parameters
        ----------
        chunk : dict[str, str]
            { doc_id -> raw text } from stream_documents()

        Returns
        -------
        dict with keys: docs_added, docs_skipped, tokens_added, vocab_size
        """
        batch_tokens  = 0
        docs_added    = 0
        docs_skipped  = 0

        for doc_id, text in chunk.items():

            # Skip duplicates — protects index integrity
            if doc_id in self.doc_lengths:
                docs_skipped += 1
                continue

            tokens = preprocess(text)
            dl     = len(tokens)

            self.doc_lengths[doc_id]  = dl
            self._total_tokens       += dl
            batch_tokens             += dl
            docs_added               += 1

            tf_counter: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counter[token] += 1

            for term, freq in tf_counter.items():
                self.index[term][doc_id] = freq

        # Only count docs actually added (v2 bug fix)
        self.N    += docs_added
        self.avgdl = self._total_tokens / self.N if self.N > 0 else 0.0

        return {
            "docs_added"  : docs_added,
            "docs_skipped": docs_skipped,
            "tokens_added": batch_tokens,
            "vocab_size"  : len(self.index),
        }

    # ── Smart incremental build driver (NEW in v3) ────────────────────────────

    def build_smart(
        self,
        filepath      : str,
        index_path    : str,
        manifest_path : str,
        chunk_size    : int  = 1000,
        force_rebuild : bool = False,
    ) -> None:
        """
        Build or update the index using manifest-based change detection.

        Decision logic
        --------------
        1. force_rebuild=True         → delete pkl + manifest → full rebuild
        2. No pkl exists              → full build from scratch
        3. pkl exists, no new docs    → skip entirely (instant)
        4. pkl exists, new docs found → load pkl → index new docs only

        This means re-running with unchanged data is free (0ms).
        Adding 1,000 new docs to a 5,000-doc index costs exactly
        1,000 preprocessing operations, not 6,000.

        Parameters
        ----------
        filepath      : str   path to documents.json
        index_path    : str   path to index.pkl
        manifest_path : str   path to index_manifest.json
        chunk_size    : int   docs per epoch (default 1000)
        force_rebuild : bool  wipe and rebuild from scratch
        """
        from Data_loader import stream_documents

        # ── Step 1: Handle force rebuild ──────────────────────────────────────
        if force_rebuild:
            print("[Index] Force rebuild requested — clearing existing index")
            for path in (index_path, manifest_path):
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[Index] Deleted: {path}")

        # ── Step 2: Load manifest to know what's already indexed ──────────────
        known_ids = self.load_manifest(manifest_path)
        print(f"[Index] Manifest: {len(known_ids):,} docs already indexed")

        # ── Step 3: Scan documents to find new ones ───────────────────────────
        print("[Index] Scanning documents for new additions...")
        all_new_docs : dict[str, str] = {}

        for chunk in stream_documents(filepath, chunk_size=chunk_size):
            for doc_id, text in chunk.items():
                if doc_id not in known_ids:
                    all_new_docs[doc_id] = text

        print(f"[Index] New documents found: {len(all_new_docs):,}")

        # ── Step 4: Nothing to do ─────────────────────────────────────────────
        if not all_new_docs:
            print("[Index] ✅ Index is up to date — nothing to rebuild")
            return

        # ── Step 5: Load existing index if it exists ──────────────────────────
        if InvertedIndex.exists(index_path) and not force_rebuild:
            print("[Index] Loading existing index to extend...")
            loaded = InvertedIndex.load(index_path)
            self.index         = loaded.index
            self.doc_lengths   = loaded.doc_lengths
            self.avgdl         = loaded.avgdl
            self.N             = loaded.N
            self._total_tokens = loaded._total_tokens

        # ── Step 6: Index only new documents in epochs ────────────────────────
        print(f"\n[Index] Indexing {len(all_new_docs):,} new docs "
              f"| chunk_size={chunk_size}")
        print(f"[Index] {'Epoch':<8} {'Added':>6} {'Total':>7} "
              f"{'Vocab':>7} {'avgdl':>7} {'Time':>6}")
        print(f"[Index] {'-'*52}")

        start_all  = time.time()
        epoch      = 0
        new_items  = list(all_new_docs.items())

        for i in range(0, len(new_items), chunk_size):
            epoch    += 1
            batch     = dict(new_items[i : i + chunk_size])
            t0        = time.time()
            stats     = self.add_batch(batch)
            elapsed   = time.time() - t0

            print(f"[Index] {epoch:<8} "
                  f"{stats['docs_added']:>6,} "
                  f"{self.N:>7,} "
                  f"{stats['vocab_size']:>7,} "
                  f"{self.avgdl:>7.1f} "
                  f"{elapsed:>5.2f}s")

        total_elapsed = time.time() - start_all
        print(f"[Index] {'-'*52}")
        self._print_stats(total_elapsed)

        # ── Step 7: Save pkl + manifest ───────────────────────────────────────
        self.save(index_path)
        self.save_manifest(manifest_path, set(self.doc_lengths.keys()))
        print(f"[Index] Manifest saved → {manifest_path} "
              f"({len(self.doc_lengths):,} doc_ids)")

    # ── Full incremental build (v2 interface, kept for compatibility) ─────────

    def build_incremental(
        self,
        filepath   : str,
        chunk_size : int = 1000,
    ) -> None:
        """
        v2 interface — build index by streaming all documents.
        Use build_smart() for manifest-protected updates.
        """
        from Data_loader import stream_documents

        print(f"[Index] Incremental build  |  chunk_size={chunk_size}")
        print(f"[Index] {'Epoch':<8} {'Docs':>6} {'Total':>7} "
              f"{'Vocab':>7} {'avgdl':>7} {'Time':>6}")
        print(f"[Index] {'-'*52}")

        start_all = time.time()
        epoch     = 0

        for chunk in stream_documents(filepath, chunk_size=chunk_size):
            epoch   += 1
            t0       = time.time()
            stats    = self.add_batch(chunk)
            elapsed  = time.time() - t0

            print(f"[Index] {epoch:<8} "
                  f"{stats['docs_added']:>6,} "
                  f"{self.N:>7,} "
                  f"{stats['vocab_size']:>7,} "
                  f"{self.avgdl:>7.1f} "
                  f"{elapsed:>5.2f}s")

        self._print_stats(time.time() - start_all)

    # ── v1 full build (kept for backward compatibility) ───────────────────────

    def build(self, documents: dict[str, str]) -> None:
        """v1 interface — index a full documents dict at once."""
        print("[Index] Building index (single batch) ...")
        start = time.time()
        self.add_batch(documents)
        self._print_stats(time.time() - start)

    # ── Disk persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize index to disk using pickle."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True
        )
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
        """Deserialize index from disk. Returns ready-to-query InvertedIndex."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Index] No index file at: {path}")

        start = time.time()
        with open(path, "rb") as f:
            payload = pickle.load(f)

        idx               = cls()
        idx.index         = defaultdict(dict, payload["index"])
        idx.doc_lengths   = payload["doc_lengths"]
        idx.avgdl         = payload["avgdl"]
        idx.N             = payload["N"]
        idx._total_tokens = payload["_total_tokens"]

        elapsed = time.time() - start
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Index] Loaded ← {path}  "
              f"({size_mb:.1f} MB  |  {idx.N:,} docs  |  {elapsed:.2f}s)")
        return idx

    @staticmethod
    def exists(path: str) -> bool:
        """Return True if a saved index file exists at path."""
        return os.path.exists(path)

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> dict[str, int]:
        """Return { doc_id -> tf } for all docs containing term."""
        return self.index.get(term, {})

    def get_df(self, term: str) -> int:
        """Document frequency — how many docs contain this term."""
        return len(self.index.get(term, {}))

    def get_idf(self, term: str) -> float:
        """
        BM25 IDF: log( (N - df + 0.5) / (df + 0.5) + 1 )
        Returns 0.0 for unknown terms.
        """
        df = self.get_df(term)
        if df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _print_stats(self, elapsed: float) -> None:
        vocab_size     = len(self.index)
        total_postings = sum(len(v) for v in self.index.values())
        df_counts      = [len(v) for v in self.index.values()]
        rare_terms     = sum(1 for d in df_counts if d == 1)
        common_terms   = sum(1 for d in df_counts if d > self.N * 0.1)

        print(f"[Index] Documents indexed : {self.N:,}")
        print(f"[Index] Vocabulary size   : {vocab_size:,} unique terms")
        print(f"[Index] Total postings    : {total_postings:,}")
        print(f"[Index] Avg doc length    : {self.avgdl:.1f} tokens")
        print(f"[Index] Rare terms (df=1) : {rare_terms:,}")
        print(f"[Index] Common terms(>10%): {common_terms:,}")
        print(f"[Index] Build time        : {elapsed:.2f}s\n")


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir      = sys.argv[1] if len(sys.argv) > 1 else "."
    doc_path      = os.path.join(data_dir, "documents.json")
    index_path    = os.path.join(data_dir, INDEX_FILENAME)
    manifest_path = os.path.join(data_dir, MANIFEST_FILENAME)

    rebuild_flag = "--rebuild" in sys.argv

    print("=" * 60)
    print("MANIFEST-BASED SMART BUILD TEST")
    print("=" * 60)

    idx = InvertedIndex()
    idx.build_smart(
        filepath      = doc_path,
        index_path    = index_path,
        manifest_path = manifest_path,
        chunk_size    = 1000,
        force_rebuild = rebuild_flag,
    )

    print("\n-- Verify loaded index --")
    idx2 = InvertedIndex.load(index_path)
    manifest = InvertedIndex.load_manifest(manifest_path)

    checks = [
        ("N",            idx.N,                   idx2.N),
        ("avgdl",        round(idx.avgdl, 4),      round(idx2.avgdl, 4)),
        ("vocab",        len(idx.index),           len(idx2.index)),
        ("manifest size",len(manifest),            idx2.N),
        ("mortgag df",   idx.get_df("mortgag"),    idx2.get_df("mortgag")),
        ("401k idf",     round(idx.get_idf("401k"),4),
                         round(idx2.get_idf("401k"),4)),
    ]

    all_ok = True
    for label, original, loaded in checks:
        ok      = original == loaded
        all_ok  = all_ok and ok
        status  = "✓" if ok else "✗ MISMATCH"
        print(f"  {label:<18} original={original}  loaded={loaded}  {status}")

    print(f"\n  All checks passed: {all_ok}")
    print(f"\nUsage:")
    print(f"  python index.py data/          # smart update (adds new docs only)")
    print(f"  python index.py data/ --rebuild # force full rebuild")