"""
Test ThreadSafeFileCache from streaming module.

Tests thread safety, LRU eviction, and proper resource cleanup.
"""

import sys
import threading
import time
from pathlib import Path

# Add code/src to path for src.ava imports
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir))


def test_thread_safe_cache_import():
    """Test that ThreadSafeFileCache imports correctly."""
    from src.ava.data.streaming import ThreadSafeFileCache
    print("✓ ThreadSafeFileCache imports successfully")


def test_basic_get_put():
    """Test basic get/put operations."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=5)

    # Put and get
    cache.put(("file1.txt", 0), "generator1")
    result = cache.get(("file1.txt", 0))
    assert result == "generator1", f"Expected 'generator1', got {result}"

    # Get non-existent key
    result = cache.get(("nonexistent.txt", 0))
    assert result is None, f"Expected None, got {result}"

    print("✓ Basic get/put operations work correctly")


def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=3)

    # Fill cache
    cache.put(("file1.txt", 0), "gen1")
    cache.put(("file2.txt", 0), "gen2")
    cache.put(("file3.txt", 0), "gen3")

    assert len(cache) == 3

    # Adding 4th should evict oldest (file1)
    cache.put(("file4.txt", 0), "gen4")

    assert len(cache) == 3
    assert cache.get(("file1.txt", 0)) is None, "file1 should have been evicted"
    assert cache.get(("file4.txt", 0)) == "gen4"

    print("✓ LRU eviction works correctly")


def test_lru_ordering():
    """Test that accessing an item moves it to end (prevents eviction)."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=3)

    cache.put(("file1.txt", 0), "gen1")
    cache.put(("file2.txt", 0), "gen2")
    cache.put(("file3.txt", 0), "gen3")

    # Access file1 (moves to end)
    cache.get(("file1.txt", 0))

    # Add file4 - should evict file2 (now oldest)
    cache.put(("file4.txt", 0), "gen4")

    assert cache.get(("file1.txt", 0)) == "gen1", "file1 should still be present"
    assert cache.get(("file2.txt", 0)) is None, "file2 should have been evicted"

    print("✓ LRU ordering (access moves to end) works correctly")


def test_clear():
    """Test clear operation."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=5)
    cache.put(("file1.txt", 0), "gen1")
    cache.put(("file2.txt", 0), "gen2")

    assert len(cache) == 2

    cache.clear()

    assert len(cache) == 0
    assert cache.get(("file1.txt", 0)) is None

    print("✓ Clear operation works correctly")


def test_evict_under_pressure():
    """Test evict_under_pressure method."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=10)

    # Fill with 10 items
    for i in range(10):
        cache.put((f"file{i}.txt", 0), f"gen{i}")

    assert len(cache) == 10

    # Evict down to 3
    cache.evict_under_pressure(3)

    assert len(cache) == 3, f"Expected 3 items, got {len(cache)}"

    print("✓ evict_under_pressure works correctly")


def test_contains():
    """Test __contains__ method."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=5)
    cache.put(("file1.txt", 0), "gen1")

    assert ("file1.txt", 0) in cache
    assert ("file2.txt", 0) not in cache

    print("✓ __contains__ works correctly")


def test_concurrent_access():
    """Test thread safety with concurrent access."""
    from src.ava.data.streaming import ThreadSafeFileCache

    cache = ThreadSafeFileCache(max_size=100)
    errors = []

    def writer(worker_id):
        try:
            for i in range(100):
                cache.put((f"file{i}.txt", worker_id), f"gen_{worker_id}_{i}")
                time.sleep(0.001)  # Small delay to increase contention
        except Exception as e:
            errors.append(f"Writer {worker_id}: {e}")

    def reader(worker_id):
        try:
            for i in range(100):
                cache.get((f"file{i}.txt", worker_id))
                time.sleep(0.001)
        except Exception as e:
            errors.append(f"Reader {worker_id}: {e}")

    # Create multiple writer and reader threads
    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader, args=(i,)))

    # Start all threads
    for t in threads:
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"Concurrent access errors: {errors}"

    print("✓ Concurrent access is thread-safe")


def test_generator_close_on_eviction():
    """Test that generators with close() method are closed on eviction."""
    from src.ava.data.streaming import ThreadSafeFileCache

    class MockGenerator:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True

    cache = ThreadSafeFileCache(max_size=2)

    gen1 = MockGenerator("gen1")
    gen2 = MockGenerator("gen2")
    gen3 = MockGenerator("gen3")

    cache.put(("file1.txt", 0), gen1)
    cache.put(("file2.txt", 0), gen2)

    # This should evict gen1
    cache.put(("file3.txt", 0), gen3)

    assert gen1.closed, "gen1 should have been closed on eviction"
    assert not gen2.closed, "gen2 should not be closed yet"
    assert not gen3.closed, "gen3 should not be closed"

    print("✓ Generators are closed on eviction")


def run_all_tests():
    """Run all tests."""
    print("\n=== ThreadSafeFileCache Tests ===\n")

    test_thread_safe_cache_import()
    test_basic_get_put()
    test_lru_eviction()
    test_lru_ordering()
    test_clear()
    test_evict_under_pressure()
    test_contains()
    test_concurrent_access()
    test_generator_close_on_eviction()

    print("\n=== All ThreadSafeFileCache tests passed! ===\n")


if __name__ == "__main__":
    run_all_tests()
