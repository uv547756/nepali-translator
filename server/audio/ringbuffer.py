"""
Thread-safe circular buffer for float32 audio samples.

Designed for the producer/consumer pattern across OS threads:
the sounddevice callback (C thread) writes; inference workers read.
Uses a Condition rather than busy-waiting so readers block cheaply.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RingBuffer:
    """Thread-safe circular buffer for float32 audio samples.

    Supports both blocking and non-blocking reads.
    Write never blocks — it overwrites old data when full
    (stale audio is worse than dropped audio for real-time inference).
    """

    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32) -> None:
        self._capacity = capacity_samples
        self._dtype = dtype
        self._buf: np.ndarray = np.zeros(capacity_samples, dtype=dtype)
        self._write_pos: int = 0
        self._read_pos: int = 0
        self._count: int = 0          # number of unread samples
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ── Write ──────────────────────────────────────────────────────────────

    def write(self, data: np.ndarray) -> int:
        """Write samples into the buffer, overwriting oldest data if full.

        Returns the number of samples actually stored (may be less than
        len(data) if data is larger than the buffer capacity).
        """
        data = np.asarray(data, dtype=self._dtype).ravel()
        n = len(data)
        if n == 0:
            return 0

        with self._not_empty:
            if n >= self._capacity:
                # Data is larger than buffer: store only the most recent samples
                data = data[-self._capacity:]
                n = self._capacity

            space = self._capacity - self._count
            if n > space:
                # Overwrite oldest: advance read pointer
                overflow = n - space
                self._read_pos = (self._read_pos + overflow) % self._capacity
                self._count -= overflow

            # Write may wrap around the end of the buffer
            tail = self._capacity - self._write_pos
            if n <= tail:
                self._buf[self._write_pos : self._write_pos + n] = data
            else:
                self._buf[self._write_pos :] = data[:tail]
                self._buf[: n - tail] = data[tail:]

            self._write_pos = (self._write_pos + n) % self._capacity
            self._count += n
            self._not_empty.notify_all()

        return n

    # ── Read ───────────────────────────────────────────────────────────────

    def read(self, n_samples: int, block: bool = True, timeout: Optional[float] = None) -> np.ndarray:
        """Read and consume up to n_samples from the buffer.

        With block=True, waits until at least one sample is available.
        With block=False, returns whatever is available immediately (may be empty).
        Returns a copy — safe to hold after subsequent writes.
        """
        with self._not_empty:
            if block:
                self._not_empty.wait_for(lambda: self._count > 0, timeout=timeout)

            available = min(n_samples, self._count)
            if available == 0:
                return np.empty(0, dtype=self._dtype)

            out = np.empty(available, dtype=self._dtype)
            tail = self._capacity - self._read_pos
            if available <= tail:
                out[:] = self._buf[self._read_pos : self._read_pos + available]
            else:
                out[:tail] = self._buf[self._read_pos :]
                out[tail:] = self._buf[: available - tail]

            self._read_pos = (self._read_pos + available) % self._capacity
            self._count -= available
            return out

    def read_all(self) -> np.ndarray:
        """Return a copy of all unread samples without consuming them."""
        with self._lock:
            if self._count == 0:
                return np.empty(0, dtype=self._dtype)

            out = np.empty(self._count, dtype=self._dtype)
            tail = self._capacity - self._read_pos
            if self._count <= tail:
                out[:] = self._buf[self._read_pos : self._read_pos + self._count]
            else:
                out[:tail] = self._buf[self._read_pos :]
                out[tail:] = self._buf[: self._count - tail]
            return out

    def snapshot(self) -> np.ndarray:
        """Alias for read_all — returns all unread data without consuming."""
        return self.read_all()

    def clear(self) -> None:
        with self._not_empty:
            self._write_pos = 0
            self._read_pos = 0
            self._count = 0

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity

    def __len__(self) -> int:
        return self.available
