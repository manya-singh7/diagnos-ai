import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from rank_bm25 import BM25Okapi

logger = logging.getLogger("diagnos_ai.retrieval")

_DEEPLINK_STOPWORDS = {
    "it", "will", "to", "and", "in", "on", "the", "a", "an", "for", "of", "with",
    "or", "by", "at", "from", "how", "what", "which", "your", "my", "is", "are",
    "be", "do", "does", "did", "let", "you", "open", "tap", "under", "per", "into",
    "then", "when", "if", "this", "that", "all", "can", "adjust", "check", "set",
}

_GENERIC_MATCH_WORDS = {
    "device", "phone", "mobile", "samsung", "galaxy", "settings", "setting",
    "options", "option", "feature", "screen", "component", "item", "hardware",
    "action", "troubleshooting", "configuration", "issue", "problem",
}

def tokenize(text: str) -> List[str]:
    """Tokenize text into content keywords stripped of stopwords and generic tokens."""
    words = re.findall(r"\b[a-z0-9'-]+\b", text.lower())
    return [w for w in words if w not in _DEEPLINK_STOPWORDS and w not in _GENERIC_MATCH_WORDS and len(w) > 2]

def get_bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

def find_catalog_path(explicit_path: Optional[Union[str, Path]] = None) -> Tuple[Path, bool]:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p, "sample" in p.name.lower()
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    real_path = root_dir / "deeplinks.json"
    if real_path.exists():
        return real_path, False

    sample_path = root_dir / "deeplinks.sample.json"
    if sample_path.exists():
        return sample_path, True

    cwd_real = Path.cwd() / "deeplinks.json"
    if cwd_real.exists():
        return cwd_real, False

    cwd_sample = Path.cwd() / "deeplinks.sample.json"
    if cwd_sample.exists():
        return cwd_sample, True

    raise FileNotFoundError("Could not find 'deeplinks.json' or 'deeplinks.sample.json'.")

class BM25Retriever:
    """
    BM25-based retriever for Samsung Galaxy Settings deeplinks.
    Indexes description, message, and qna_description metadata fields.
    Never indexes URI strings or domains.
    """
    def __init__(self, catalog_path: Optional[Union[str, Path]] = None):
        self.catalog_path, self.is_sample = find_catalog_path(catalog_path)
        if self.is_sample:
            warn_msg = (
                "[WARNING] 'deeplinks.json' not found in project root. "
                "Falling back to 'deeplinks.sample.json'. "
                "Sample data is in use — results are not final numbers."
            )
            logger.warning(warn_msg)
            print(warn_msg)
        
        self.catalog = self._load_catalog(self.catalog_path)
        # Filter out dummy_positive from indexing
        self.indexed_docs = [
            item for item in self.catalog if item.get("deeplink") != "bixby://dummy_positive"
        ]
        
        self.tokenized_corpus = [
            tokenize(
                f"{item.get('description', '')} {item.get('message', '')} {item.get('qna_description', '')}"
            )
            for item in self.indexed_docs
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _load_catalog(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieves top-k catalog entries matching the query with normalized relevance scores.
        Returns List of (catalog_item, score) tuples sorted descending by score.
        """
        if not self.bm25 or not self.indexed_docs:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_bigrams = get_bigrams(q_tokens)
        unique_q_tokens = set(q_tokens)
        raw_scores = self.bm25.get_scores(q_tokens)

        scored_results: List[Tuple[Dict[str, Any], float]] = []

        for item, doc_tokens, raw_score in zip(self.indexed_docs, self.tokenized_corpus, raw_scores):
            cand_text = f"{item.get('description', '')} {item.get('message', '')} {item.get('qna_description', '')}".lower()
            matched_tokens = unique_q_tokens & set(doc_tokens)
            
            if not matched_tokens:
                scored_results.append((item, 0.0))
                continue

            overlap_ratio = len(matched_tokens) / len(unique_q_tokens)
            has_bigram = any(bg in cand_text for bg in q_bigrams)
            
            # Combine term coverage ratio and BM25 signal + bigram precision
            bigram_boost = 0.3 if has_bigram else 0.0
            bm25_component = min(0.3, (raw_score / 10.0) * 0.3) if not has_bigram else 0.0
            score = min(1.0, (overlap_ratio * 0.7) + bigram_boost + bm25_component)
            
            if score >= min_score:
                scored_results.append((item, round(score, 4)))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def get_top_match(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        matches = self.retrieve(query, top_k=1)
        if matches:
            return matches[0]
        return None, 0.0


_RETRIEVER_INSTANCE: Optional[BM25Retriever] = None

def get_retriever(catalog_path: Optional[Union[str, Path]] = None, force_reload: bool = False) -> BM25Retriever:
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None or force_reload:
        _RETRIEVER_INSTANCE = BM25Retriever(catalog_path=catalog_path)
    return _RETRIEVER_INSTANCE
