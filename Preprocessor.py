"""
preprocessor.py
---------------
Text preprocessing pipeline for the IR system.

Pipeline (applied identically to documents AND queries):
  1. Lowercase
  2. Tokenize       — keep only alphanumeric tokens (strip punctuation)
  3. Stop word removal
  4. Porter Stemming

Requires:
    pip install nltk

No NLTK corpus downloads needed — stopwords are bundled below.
"""

import re
from nltk.stem import PorterStemmer

# ── Bundled English stop words (standard IR set) ──────────────────────────────
# Sourced from the NLTK English stopwords list — bundled here so no
# nltk.download() call is needed on the user's machine.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't",
    "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
}

# ── Stemmer (singleton — reuse across all calls) ──────────────────────────────
_stemmer = PorterStemmer()


def tokenize(text: str) -> list[str]:
    """
    Lowercase and split text into alphanumeric tokens.
    Strips punctuation and standalone digits under 2 chars.

    Example
    -------
    >>> tokenize("Buying from an aggressive salesperson!")
    ['buying', 'from', 'an', 'aggressive', 'salesperson']
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Remove common English stop words from token list.

    Example
    -------
    >>> remove_stopwords(['buying', 'from', 'an', 'aggressive', 'salesperson'])
    ['buying', 'aggressive', 'salesperson']
    """
    return [t for t in tokens if t not in STOP_WORDS]


def stem(tokens: list[str]) -> list[str]:
    """
    Apply Porter Stemming to reduce tokens to their root form.

    Example
    -------
    >>> stem(['buying', 'aggressive', 'salesperson'])
    ['buy', 'aggress', 'salesperson']
    """
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline: tokenize → remove stopwords → stem.

    This is the main function used by the indexer and query processor.

    Parameters
    ----------
    text : str
        Raw document or query text.

    Returns
    -------
    list[str]
        Cleaned, stemmed tokens ready for indexing or matching.

    Example
    -------
    >>> preprocess("Should you always max out contributions to your 401k?")
    ['alway', 'max', 'contribut', '401k']
    """
    if not text or not text.strip():
        return []

    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens


# ── Quick sanity check when run directly ─────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "Buying from an aggressive salesperson",
        "Should you always max out contributions to your 401k?",
        "Why does short selling require borrowing?",
        "15 year mortgage vs 30 year paid off in 15",
        "",                          # edge case: empty string
        "!!! ???",                   # edge case: punctuation only
    ]

    print("=" * 60)
    print(f"{'Input':<45} → Tokens")
    print("=" * 60)
    for text in test_cases:
        tokens = preprocess(text)
        display = (text[:42] + "...") if len(text) > 45 else text
        print(f"{display:<45} → {tokens}")