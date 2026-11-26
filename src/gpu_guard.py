"""Admission control for GPU work, in two levels, because one is not enough.

Five runs in this project have died of CUDA OOM. Every one of them had a lock
around the *inference call* -- and that lock was never the problem. Model weights
are loaded once and stay resident for the process's whole life, outside any
per-call lock: four workers each holding a 496 M-parameter U-Net have already
spent 8 GB before a single forward pass. Serialising the forward passes cannot
help with memory that was allocated before them.

So there are two different things to control and they need different mechanisms:

1. **Admission** (`gpu_slot`) -- how many processes may hold models resident at
   all. Acquired once at start-up, before anything is loaded, and held for the
   process lifetime.
2. **Peak serialisation** (the existing `gpu_lock`) -- how many may be in a
   forward pass at once. Acquired around the call.

Admission is also the level that has to notice **other people's jobs**. A
counting semaphore over this project's own workers is blind to an unrelated
process holding several GB, which is exactly what happened most recently. So
admission additionally waits on measured free memory and keeps a headroom
margin, rather than trusting its own bookkeeping.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import time

LOCK_DIR = "/tmp/claude-1000/gpu_slots"
DEFAULT_SLOTS = 2
HEADROOM_MB = 1800          # left for processes this module does not know about
POLL_S = 5.0


def free_mb():
    """Free GPU memory in MB, or None if it cannot be measured."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


@contextlib.contextmanager
def gpu_slot(need_mb=3000, slots=DEFAULT_SLOTS, headroom_mb=HEADROOM_MB,
             timeout=None, label=""):
    """Hold one of `slots` admission tokens, and wait for room to actually exist.

    Two conditions, both required. The token bounds how many of *our* processes
    load models; the memory poll bounds the total, including anything else on
    the card. A token alone is what failed before.

    Call this once, before importing weights -- not around inference.
    """
    os.makedirs(LOCK_DIR, exist_ok=True)
    t0 = time.time()
    fh = None
    try:
        while True:
            for i in range(slots):
                f = open(os.path.join(LOCK_DIR, f"slot{i}.lock"), "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    f.close()
                    continue
                fh = f
                break
            if fh is not None:
                # token held; now wait until the card can actually take us
                while True:
                    fm = free_mb()
                    if fm is None or fm >= need_mb + headroom_mb:
                        break
                    if timeout is not None and time.time() - t0 > timeout:
                        raise TimeoutError(
                            f"gpu_slot({label}): {fm} MB free, need "
                            f"{need_mb}+{headroom_mb}, waited {timeout:.0f}s")
                    time.sleep(POLL_S)
                break
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"gpu_slot({label}): no free slot after {timeout:.0f}s")
            time.sleep(POLL_S)
        yield
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh, fcntl.LOCK_UN)
            finally:
                fh.close()
