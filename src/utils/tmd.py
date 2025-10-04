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


def encode_tmd16(domain: int, task: int, modifier: int) -> np.ndarray:
    """Return a stable 16D float vector derived from categorical codes."""
    assert 0 <= domain <= 0xF and 0 <= task <= 0x1F and 0 <= modifier <= 0x3F

    bits = np.concatenate([
        _code_bits(domain, 4),
        _code_bits(task, 5),
        _code_bits(modifier, 6),
    ])  # (15,)

    vec = _PROJ @ bits
    try:
        from .norms import l2_normalize
    except ImportError:  # pragma: no cover
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from norms import l2_normalize  # type: ignore

    return l2_normalize(vec)


def format_tmd_code(bits_or_codes: int | dict | None) -> str:
    """Format TMD code from various input formats.

    Args:
        bits_or_codes: Either packed TMD bits (int), or dict with 'tmd_bits'/'domain_code'/'task_code'/'modifier_code'

    Returns:
        Formatted D.T.M string like "2.0.27" or "0.0.0" if invalid
    """
    if isinstance(bits_or_codes, int):
        try:
            d, t, m = unpack_tmd(bits_or_codes)
            return f"{d}.{t}.{m}"
        except Exception:
            return "0.0.0"
    elif isinstance(bits_or_codes, dict):
        # Try tmd_bits first
        tmd_bits = bits_or_codes.get("tmd_bits")
        if tmd_bits is not None:
            try:
                d, t, m = unpack_tmd(int(tmd_bits))
                return f"{d}.{t}.{m}"
            except Exception:
                pass

        # Fallback to individual codes
        d = bits_or_codes.get("domain_code")
        t = bits_or_codes.get("task_code")
        m = bits_or_codes.get("modifier_code")
        if d is not None and t is not None and m is not None:
            try:
                return f"{int(d)}.{int(t)}.{int(m)}"
            except Exception:
                return "0.0.0"

        # Final fallback to existing tmd_code
        return bits_or_codes.get("tmd_code") or "0.0.0"

    return "0.0.0"
