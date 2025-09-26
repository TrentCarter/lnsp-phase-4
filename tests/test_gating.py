"""
Tests for CPESH gating functionality
"""

import pytest
from src.utils.gating import CPESHGateConfig, apply_lane_overrides, should_use_cpesh


class TestCPESHGating:
    """Test CPESH gating logic."""

    def test_gate_pass_basic(self):
        """Test basic gating pass case."""
        cfg = CPESHGateConfig()
        entry = {"quality": 0.90, "cosine": 0.56}
        assert should_use_cpesh(entry, cfg) is True

    def test_gate_fail_quality(self):
        """Test gating fails when quality is too low."""
        cfg = CPESHGateConfig()
        entry = {"quality": 0.80, "cosine": 0.70}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_fail_cos(self):
        """Test gating fails when cosine similarity is too low."""
        cfg = CPESHGateConfig()
        entry = {"quality": 0.90, "cosine": 0.50}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_fail_insufficient_evidence(self):
        """Test gating fails when insufficient_evidence flag is set."""
        cfg = CPESHGateConfig()
        entry = {"quality": 0.95, "cosine": 0.80, "insufficient_evidence": True}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_pass_boundary_values(self):
        """Test gating with exact boundary values."""
        cfg = CPESHGateConfig(q_min=0.82, cos_min=0.55)
        entry = {"quality": 0.82, "cosine": 0.55}
        assert should_use_cpesh(entry, cfg) is True

    def test_gate_fail_boundary_values(self):
        """Test gating fails with values below boundary."""
        cfg = CPESHGateConfig(q_min=0.82, cos_min=0.55)
        entry = {"quality": 0.81, "cosine": 0.54}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_none_entry(self):
        """Test gating with None entry."""
        cfg = CPESHGateConfig()
        assert should_use_cpesh(None, cfg) is False

    def test_gate_missing_fields(self):
        """Test gating with missing quality or cosine fields."""
        cfg = CPESHGateConfig()

        # Missing quality
        entry = {"cosine": 0.60}
        assert should_use_cpesh(entry, cfg) is False

        # Missing cosine
        entry = {"quality": 0.90}
        assert should_use_cpesh(entry, cfg) is False

        # Both missing
        entry = {}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_zero_values(self):
        """Test gating with zero values."""
        cfg = CPESHGateConfig()
        entry = {"quality": 0.0, "cosine": 0.0}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_negative_values(self):
        """Test gating with negative values."""
        cfg = CPESHGateConfig()
        entry = {"quality": -0.1, "cosine": -0.1}
        assert should_use_cpesh(entry, cfg) is False

    def test_gate_perfect_scores(self):
        """Test gating with perfect scores."""
        cfg = CPESHGateConfig()
        entry = {"quality": 1.0, "cosine": 1.0}
        assert should_use_cpesh(entry, cfg) is True


class TestLaneOverrides:
    """Test lane-specific configuration overrides."""

    def test_lane_override_factoid(self):
        """Test L1_FACTOID lane override."""
        base = CPESHGateConfig(q_min=0.82)
        lane_cfg = apply_lane_overrides(base, "L1_FACTOID")
        assert lane_cfg.q_min >= 0.85

    def test_lane_override_nonexistent(self):
        """Test override with non-existent lane."""
        base = CPESHGateConfig(q_min=0.82, cos_min=0.55)
        lane_cfg = apply_lane_overrides(base, "NONEXISTENT_LANE")
        assert lane_cfg.q_min == 0.82
        assert lane_cfg.cos_min == 0.55

    def test_lane_override_none_lane(self):
        """Test override with None lane."""
        base = CPESHGateConfig(q_min=0.82, cos_min=0.55)
        lane_cfg = apply_lane_overrides(base, None)
        assert lane_cfg.q_min == 0.82
        assert lane_cfg.cos_min == 0.55

    def test_lane_override_partial(self):
        """Test partial lane overrides."""
        base = CPESHGateConfig(q_min=0.82, cos_min=0.55, nprobe_cpesh=8)
        overrides = {"L1_FACTOID": {"q_min": 0.85}}  # Only override q_min
        base.lane_overrides = overrides

        lane_cfg = apply_lane_overrides(base, "L1_FACTOID")
        assert lane_cfg.q_min == 0.85
        assert lane_cfg.cos_min == 0.55  # Should remain unchanged
        assert lane_cfg.nprobe_cpesh == 8  # Should remain unchanged

    def test_lane_override_all_params(self):
        """Test overriding all parameters for a lane."""
        base = CPESHGateConfig(q_min=0.82, cos_min=0.55, nprobe_cpesh=8, nprobe_fallback=16)
        overrides = {"L2_GRAPH": {"q_min": 0.88, "cos_min": 0.60, "nprobe_cpesh": 12, "nprobe_fallback": 20}}
        base.lane_overrides = overrides

        lane_cfg = apply_lane_overrides(base, "L2_GRAPH")
        assert lane_cfg.q_min == 0.88
        assert lane_cfg.cos_min == 0.60
        assert lane_cfg.nprobe_cpesh == 12
        assert lane_cfg.nprobe_fallback == 20

    def test_multiple_lanes(self):
        """Test multiple lane configurations."""
        base = CPESHGateConfig(q_min=0.82)
        overrides = {
            "L1_FACTOID": {"q_min": 0.85},
            "L2_GRAPH": {"q_min": 0.80},
            "L3_SYNTH": {"q_min": 0.90}
        }
        base.lane_overrides = overrides

        # Test each lane
        l1_cfg = apply_lane_overrides(base, "L1_FACTOID")
        assert l1_cfg.q_min == 0.85

        l2_cfg = apply_lane_overrides(base, "L2_GRAPH")
        assert l2_cfg.q_min == 0.80

        l3_cfg = apply_lane_overrides(base, "L3_SYNTH")
        assert l3_cfg.q_min == 0.90

    def test_lane_override_with_gating_decision(self):
        """Test that lane overrides affect gating decisions."""
        base = CPESHGateConfig(q_min=0.82, cos_min=0.55)
        overrides = {"L1_FACTOID": {"q_min": 0.90}}  # Stricter quality requirement
        base.lane_overrides = overrides

        # Entry that would pass base config but fail L1_FACTOID config
        entry = {"quality": 0.85, "cosine": 0.60}

        base_cfg = apply_lane_overrides(base, None)
        assert should_use_cpesh(entry, base_cfg) is True  # Should pass base config

        l1_cfg = apply_lane_overrides(base, "L1_FACTOID")
        assert should_use_cpesh(entry, l1_cfg) is False  # Should fail L1_FACTOID config
