"""
Provider-agnostic API key rotator with round-robin selection and automatic
failover on rate limiting. Shared by gemini_engine.py and groq_engine.py so
each provider's key pool rotates independently but identically.
"""

import threading
import time


class KeyRotator:
    """Round-robin API key rotation with automatic failover on rate limit."""

    def __init__(self, keys: list[str]):
        self._keys = [k for k in keys if k and len(k) > 10]
        if not self._keys:
            raise ValueError("No valid API keys provided")
        self._index = 0
        self._lock = threading.Lock()
        self._failed = {}  # key -> timestamp when it was rate-limited

    @property
    def count(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """Get the next available key, skipping recently rate-limited ones."""
        with self._lock:
            now = time.time()
            # Try each key once
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)

                # Skip if rate-limited in last 60 seconds
                if key in self._failed and now - self._failed[key] < 60:
                    continue
                return key

            # All keys rate-limited — return least recently failed and reset index past it
            oldest_key = min(self._failed, key=self._failed.get, default=self._keys[0])
            if oldest_key in self._keys:
                self._index = (self._keys.index(oldest_key) + 1) % len(self._keys)
            return oldest_key

    def mark_failed(self, key: str):
        """Mark a key as rate-limited."""
        with self._lock:
            self._failed[key] = time.time()

    def mark_success(self, key: str):
        """Clear rate-limit flag on success."""
        with self._lock:
            self._failed.pop(key, None)
