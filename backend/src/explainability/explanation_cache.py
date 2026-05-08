"""
Explanation Cache System
========================
Stores and retrieves pre-generated LLM explanations to avoid runtime API calls.

Usage:
    # Generate and cache explanations locally
    python -m backend.src.explainability.explanation_cache generate

    # Use cached explanations in Streamlit
    from backend.src.explainability.explanation_cache import ExplanationCache
    cache = ExplanationCache()
    explanation = cache.get_explanation("LP001003", audience="technical")
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ExplanationCache:
    """Manages cached LLM explanations"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize explanation cache.

        Args:
            cache_dir: Directory to store cached explanations.
                      Defaults to .resources/data/cached_explanations/
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            cache_dir = project_root / ".resources" / "data" / "cached_explanations"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "explanations.json"

        # Load existing cache
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cached explanations from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False)

    def get_explanation(self,
                       application_id: str,
                       audience: str = "technical",
                       use_concepts: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get cached explanation for an application.

        Args:
            application_id: Application ID (e.g., "LP001003")
            audience: Explanation audience ("technical", "business", "customer")
            use_concepts: Whether concepts were used in explanation

        Returns:
            Cached explanation dict or None if not found
        """
        cache_key = self._make_key(application_id, audience, use_concepts)
        return self._cache.get(cache_key)

    def set_explanation(self,
                       application_id: str,
                       audience: str,
                       use_concepts: bool,
                       explanation: Dict[str, Any]):
        """
        Cache an explanation.

        Args:
            application_id: Application ID
            audience: Explanation audience
            use_concepts: Whether concepts were used
            explanation: Full explanation dict from LLM
        """
        cache_key = self._make_key(application_id, audience, use_concepts)

        # Add metadata
        explanation['_cached_at'] = datetime.now().isoformat()
        explanation['_cache_key'] = cache_key

        self._cache[cache_key] = explanation
        self._save_cache()

        print(f"[CACHE] Saved: {cache_key}")

    def has_explanation(self,
                       application_id: str,
                       audience: str = "technical",
                       use_concepts: bool = True) -> bool:
        """Check if explanation exists in cache"""
        cache_key = self._make_key(application_id, audience, use_concepts)
        return cache_key in self._cache

    def _make_key(self, application_id: str, audience: str, use_concepts: bool) -> str:
        """Generate cache key"""
        return f"{application_id}_{audience}_concepts-{use_concepts}"

    def get_all_cached_ids(self) -> list:
        """Get list of all cached application IDs"""
        ids = set()
        for key in self._cache.keys():
            app_id = key.split('_')[0]
            ids.add(app_id)
        return sorted(list(ids))

    def clear(self):
        """Clear all cached explanations"""
        self._cache = {}
        self._save_cache()
        print("[CACHE] Cleared all cached explanations")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_cached': len(self._cache),
            'application_ids': len(self.get_all_cached_ids()),
            'cache_file': str(self.cache_file),
            'cache_size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        print("Use .development/scripts/generate_cached_explanations.py to pre-generate explanations")
    else:
        # Show cache stats
        cache = ExplanationCache()
        stats = cache.stats()
        print("\n=== Explanation Cache Stats ===")
        print(f"Total cached explanations: {stats['total_cached']}")
        print(f"Unique application IDs: {stats['application_ids']}")
        print(f"Cache file: {stats['cache_file']}")
        print(f"Cache size: {stats['cache_size_bytes']} bytes")
        print(f"\nCached IDs: {', '.join(cache.get_all_cached_ids())}")
