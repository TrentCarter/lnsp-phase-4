from __future__ import annotations
from typing import Tuple
import numpy as np

# Bit layout (uint16): [15..12]=Domain(4b) | [11..7]=Task(5b) | [6..1]=Modifier(6b) | [0]=spare
# Ranges are frozen per docs/enums.md (Domain 0-15, Task 0-31, Modifier 0-63).


def pack_tmd(domain: int, task: int, modifier: int) -> int:
    """Pack TMD codes into uint16 bit field.
    Domain: 4 bits (0-15), Task: 5 bits (0-31), Modifier: 6 bits (0-63)
    """
    assert 0 <= domain <= 0xF, f"domain {domain} out of range [0, 15]"
    assert 0 <= task <= 0x1F, f"task {task} out of range [0, 31]"
    assert 0 <= modifier <= 0x3F, f"modifier {modifier} out of range [0, 63]"
    return (domain << 12) | (task << 7) | (modifier << 1)


def unpack_tmd(bits: int) -> Tuple[int, int, int]:
    """Unpack uint16 bit field into (domain, task, modifier) codes."""
    domain = (bits >> 12) & 0xF
    task = (bits >> 7) & 0x1F
    modifier = (bits >> 1) & 0x3F
    return domain, task, modifier


def lane_index_from_bits(bits: int) -> int:
    """Map tmd_bits → lane_index (0..32767). Direct mapping with 1-bit spare."""
    return (bits >> 1) & 0x7FFF


# Deterministic 16D projection from (domain, task, modifier)
# This substitutes for a learned 16D embedding in LEAN mode.
# It hashes codes into a stable 16x(4+5+6)=16x15 sparse code projected to 16D.

_rng = np.random.default_rng(seed=1337)
_PROJ = _rng.standard_normal((16, 15)).astype(np.float32)


def _code_bits(value: int, width: int) -> np.ndarray:
    bits = [(value >> shift) & 1 for shift in range(width)]
    return np.array(bits[::-1], dtype=np.float32)


def tmd16_deterministic(domain: int, task: int, modifier: int) -> np.ndarray:
    """Return a stable 16D float vector derived from categorical codes."""
    assert 0 <= domain <= 0xF and 0 <= task <= 0x1F and 0 <= modifier <= 0x3F

    bits = np.concatenate([
        _code_bits(domain, 4),
        _code_bits(task, 5),
        _code_bits(modifier, 6),
    ])  # (15,)

    vec = _PROJ @ bits
    try:
        from .utils.norms import l2_normalize
    except ImportError:  # pragma: no cover
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from utils.norms import l2_normalize  # type: ignore
    return l2_normalize(vec.reshape(1, -1)).astype(np.float32).reshape(-1)
