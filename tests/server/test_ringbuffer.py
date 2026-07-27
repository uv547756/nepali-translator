"""Unit tests for the lock-free ring buffer."""

import threading
import numpy as np
import pytest

from server.audio.ringbuffer import RingBuffer


def test_basic_write_read():
    buf = RingBuffer(1024)
    data = np.ones(100, dtype=np.float32)
    buf.write(data)
    out = buf.read(100, block=False)
    assert len(out) == 100
    np.testing.assert_array_equal(out, data)


def test_partial_read():
    buf = RingBuffer(1024)
    buf.write(np.ones(200, dtype=np.float32))
    out = buf.read(50, block=False)
    assert len(out) == 50
    assert buf.available == 150


def test_overflow_drops_oldest():
    buf = RingBuffer(100)
    old = np.zeros(60, dtype=np.float32)
    new = np.ones(60, dtype=np.float32)
    buf.write(old)
    buf.write(new)          # should overwrite old partially
    assert buf.available == 100
    out = buf.read(100, block=False)
    # The tail (new data) should be present
    assert out[-1] == 1.0


def test_snapshot_nondestructive():
    buf = RingBuffer(512)
    data = np.arange(50, dtype=np.float32)
    buf.write(data)
    snap = buf.snapshot()
    assert len(snap) == 50
    assert buf.available == 50   # not consumed


def test_clear():
    buf = RingBuffer(512)
    buf.write(np.ones(100, dtype=np.float32))
    buf.clear()
    assert buf.available == 0
    out = buf.read(10, block=False)
    assert len(out) == 0


def test_wrap_around():
    buf = RingBuffer(100)
    buf.write(np.zeros(80, dtype=np.float32))
    buf.read(60, block=False)          # consume 60, leaving 20 at position 60
    extra = np.ones(50, dtype=np.float32)
    buf.write(extra)                   # writes wrap around: 20+50=70 available
    assert buf.available == 70


def test_threadsafe_producer_consumer():
    buf = RingBuffer(4096)
    TOTAL = 10000
    results: list[float] = []

    def producer():
        for i in range(TOTAL):
            buf.write(np.array([float(i)], dtype=np.float32))

    def consumer():
        seen = 0
        while seen < TOTAL:
            chunk = buf.read(1, block=True, timeout=5.0)
            if len(chunk) > 0:
                results.append(float(chunk[0]))
                seen += 1

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)
    t_cons.start()
    t_prod.start()
    t_prod.join()
    t_cons.join()

    assert len(results) == TOTAL
