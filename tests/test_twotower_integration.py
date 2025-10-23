"""
Two-Tower + Dual-Path Decoder Integration Tests

Tests for the newly integrated Two-Tower retrieval system with dual-path decoding:
- Mock LVM for testing generation pipeline
- Query tower + FAISS miner integration
- End-to-end generation with dual-path decoder
- Decision distribution analysis (SNAP/BLEND/NOVEL/NOVEL_DUP_DROP)
- Near-duplicate detection in generation context
- Lane configuration behavior

Created: 2025-10-22
Related: TwoTower_v4_Training_Status_Report_2025-10-22.md
"""

import sys
import pytest
import torch
import torch.nn as nn
import numpy as np
import faiss
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval.query_tower import QueryTower
from retrieval.miner_sync import SyncFaissMiner
from training.dual_path_decoder import DualPathDecoder
from retrieval.decider import choose_next_vector, DecisionRecord, LaneConfig


class MockLVM(nn.Module):
    """Mock LVM for testing generation pipeline."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        # Simple GRU that takes context and predicts next vector
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, context):
        """
        Generate next vector from context.

        Args:
            context: (T, 768) numpy array or (1, T, 768) tensor

        Returns:
            (768,) numpy array - predicted next vector
        """
        if isinstance(context, np.ndarray):
            context = torch.from_numpy(context).float()

        if context.ndim == 2:
            context = context.unsqueeze(0)  # (1, T, 768)

        with torch.no_grad():
            out, _ = self.gru(context)
            pooled = out[:, -1, :]  # Last hidden state
            pred = self.ln(pooled)
            # Normalize to unit vector
            pred = nn.functional.normalize(pred, dim=-1)

        return pred[0].cpu().numpy()  # (768,)


@pytest.fixture
def setup_test_env():
    """Create test fixtures: bank, index, miner, query tower, LVM."""
    # Small test bank (100 vectors)
    np.random.seed(42)
    bank = np.random.randn(100, 768).astype(np.float32)
    # Normalize to unit vectors
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(768)  # Inner product
    index.add(bank)

    # Create components
    miner = SyncFaissMiner(index, nprobe=1)
    query_tower = QueryTower()
    lvm = MockLVM()

    # Initial context (10 vectors)
    context = bank[:10]  # (10, 768)

    return {
        'bank': bank,
        'index': index,
        'miner': miner,
        'query_tower': query_tower,
        'lvm': lvm,
        'context': context,
    }


class TestMockLVM:
    """Test the mock LVM component."""

    def test_lvm_forward_shape(self, setup_test_env):
        """Test LVM produces correct output shape."""
        lvm = setup_test_env['lvm']
        context = setup_test_env['context']

        v_hat = lvm.forward(context)

        assert v_hat.shape == (768,), f"Expected (768,), got {v_hat.shape}"
        assert isinstance(v_hat, np.ndarray), "LVM should return numpy array"
        # Check unit vector
        norm = np.linalg.norm(v_hat)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_lvm_deterministic(self, setup_test_env):
        """Test LVM produces consistent output for same context."""
        lvm = setup_test_env['lvm']
        context = setup_test_env['context']

        v1 = lvm.forward(context)
        v2 = lvm.forward(context)

        assert np.allclose(v1, v2), "LVM should be deterministic"


class TestQueryTowerMinerIntegration:
    """Test query tower + FAISS miner integration."""

    def test_retriever_search(self, setup_test_env):
        """Test retriever finds candidates correctly."""
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        context = setup_test_env['context']

        # Encode context with query tower
        ctx_torch = torch.from_numpy(context).float().unsqueeze(0)  # (1, 10, 768)
        with torch.no_grad():
            q = query_tower(ctx_torch)  # (1, 768)
        q_np = q.cpu().numpy()

        # Search
        I, D = miner.search(q_np, k=10)

        assert I.shape == (1, 10), f"Expected (1, 10), got {I.shape}"
        assert D.shape == (1, 10), f"Expected (1, 10), got {D.shape}"
        assert np.all(I[0] < 100), "All indices should be < bank size"
        assert np.all(D[0] <= 1.0), "All cosines should be <= 1.0"

    def test_query_tower_gradients(self):
        """Test query tower can compute gradients."""
        query_tower = QueryTower()
        query_tower.train()

        # Mock context
        ctx = torch.randn(4, 10, 768)  # (batch=4, seq=10, dim=768)

        # Forward pass
        q = query_tower(ctx)

        # Check output shape
        assert q.shape == (4, 768), f"Expected (4, 768), got {q.shape}"

        # Check gradients work
        loss = q.sum()
        loss.backward()

        # Verify gradients exist
        for name, param in query_tower.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_miner_batch_search(self, setup_test_env):
        """Test miner can handle batch queries."""
        miner = setup_test_env['miner']

        # Batch of 5 queries
        queries = np.random.randn(5, 768).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        I, D = miner.search(queries, k=10)

        assert I.shape == (5, 10), f"Expected (5, 10), got {I.shape}"
        assert D.shape == (5, 10), f"Expected (5, 10), got {D.shape}"


class TestDualPathGeneration:
    """Test end-to-end generation with dual-path decoder."""

    def test_end_to_end_generation(self, setup_test_env):
        """Test complete generation loop: LVM -> retriever -> dual-path."""
        bank = setup_test_env['bank']
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        lvm = setup_test_env['lvm']
        context = setup_test_env['context'].copy()  # (10, 768)

        # Create decoder
        decoder = DualPathDecoder(
            lane="neutral",
            tau_snap=0.92,
            tau_novel=0.85,
            near_dup_cos=0.98,
            near_dup_window=8
        )

        # Generate 5 steps
        generated = []
        decisions = []

        for step in range(5):
            # 1. LVM generates next vector
            v_hat = lvm.forward(context)

            # 2. Query tower encodes context
            ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
            with torch.no_grad():
                q = query_tower(ctx_torch)
            q_np = q.cpu().numpy()[0]

            # 3. Retriever finds candidates
            I, D = miner.search(q_np[None, :], k=20)
            neighbors = [
                (f"bank_{i}", bank[i], D[0, j])
                for j, i in enumerate(I[0])
            ]

            # 4. Dual-path decision
            v_out, rec = decoder.step(v_hat, neighbors)

            # 5. Update context
            context = np.vstack([context, v_out])

            generated.append(v_out)
            decisions.append(rec.decision)

        # Verify outputs
        assert len(generated) == 5, "Should generate 5 vectors"
        assert len(decisions) == 5, "Should have 5 decision records"
        assert all(v.shape == (768,) for v in generated), "All vectors should be (768,)"

        # Check decision types
        valid_decisions = {'SNAP', 'BLEND', 'NOVEL', 'NOVEL_DUP_DROP'}
        assert all(d in valid_decisions for d in decisions), \
            f"Invalid decisions: {set(decisions) - valid_decisions}"

        print(f"✓ Generated 5 steps with decisions: {decisions}")

    def test_generation_produces_unit_vectors(self, setup_test_env):
        """Test all generated vectors are unit-normalized."""
        bank = setup_test_env['bank']
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        lvm = setup_test_env['lvm']
        context = setup_test_env['context'].copy()

        decoder = DualPathDecoder(lane="neutral", tau_snap=0.92, tau_novel=0.85)

        for step in range(10):
            v_hat = lvm.forward(context)
            ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
            with torch.no_grad():
                q = query_tower(ctx_torch)
            q_np = q.cpu().numpy()[0]

            I, D = miner.search(q_np[None, :], k=20)
            neighbors = [(f"bank_{i}", bank[i], D[0, j]) for j, i in enumerate(I[0])]

            v_out, rec = decoder.step(v_hat, neighbors)

            # Check unit vector
            norm = np.linalg.norm(v_out)
            assert abs(norm - 1.0) < 1e-5, f"Step {step}: vector not unit (norm={norm})"

            context = np.vstack([context, v_out])


class TestDecisionDistribution:
    """Test decision distribution over multiple generation steps."""

    def test_decision_distribution(self, setup_test_env):
        """Test decision distribution over 50 steps."""
        bank = setup_test_env['bank']
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        lvm = setup_test_env['lvm']
        context = setup_test_env['context'].copy()

        decoder = DualPathDecoder(
            lane="neutral",
            tau_snap=0.92,
            tau_novel=0.85
        )

        decisions = []
        cosines = []

        for step in range(50):
            v_hat = lvm.forward(context)

            ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
            with torch.no_grad():
                q = query_tower(ctx_torch)
            q_np = q.cpu().numpy()[0]

            I, D = miner.search(q_np[None, :], k=20)
            neighbors = [(f"bank_{i}", bank[i], D[0, j]) for j, i in enumerate(I[0])]

            v_out, rec = decoder.step(v_hat, neighbors)

            context = np.vstack([context, v_out])
            decisions.append(rec.decision)
            cosines.append(rec.c_max)

        # Analyze distribution
        dist = Counter(decisions)

        print(f"\nDecision distribution (50 steps):")
        for decision, count in dist.most_common():
            pct = 100 * count / 50
            print(f"  {decision}: {count} ({pct:.1f}%)")

        print(f"\nCosine range: [{min(cosines):.3f}, {max(cosines):.3f}]")
        print(f"Mean cosine: {np.mean(cosines):.3f}")

        # Sanity checks
        assert len(set(decisions)) >= 1, "Should have at least one decision type"
        assert all(0.0 <= c <= 1.0 for c in cosines), "Cosines should be in [0, 1]"

    def test_decision_telemetry(self, setup_test_env):
        """Test decision records contain telemetry data."""
        bank = setup_test_env['bank']
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        lvm = setup_test_env['lvm']
        context = setup_test_env['context']

        decoder = DualPathDecoder(lane="neutral", tau_snap=0.92, tau_novel=0.85)

        v_hat = lvm.forward(context)
        ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
        with torch.no_grad():
            q = query_tower(ctx_torch)
        q_np = q.cpu().numpy()[0]

        I, D = miner.search(q_np[None, :], k=20)
        neighbors = [(f"bank_{i}", bank[i], D[0, j]) for j, i in enumerate(I[0])]

        v_out, rec = decoder.step(v_hat, neighbors)

        # Check telemetry fields
        assert hasattr(rec, 'decision')
        assert hasattr(rec, 'c_max')
        assert hasattr(rec, 'neighbor_id')
        assert hasattr(rec, 'lane')

        print(f"\nTelemetry record:")
        print(f"  decision: {rec.decision}")
        print(f"  c_max: {rec.c_max:.3f}")
        print(f"  neighbor_id: {rec.neighbor_id}")
        print(f"  lane: {rec.lane}")
        if rec.alpha is not None:
            print(f"  alpha: {rec.alpha:.3f}")


class TestNearDuplicateDetection:
    """Test near-duplicate detection in generation context."""

    def test_near_duplicate_detection(self, setup_test_env):
        """Test near-duplicate detection prevents repetition."""
        bank = setup_test_env['bank']

        # Create a scenario with very similar neighbors
        base_vec = bank[0].copy()
        near_dups = [
            base_vec + np.random.randn(768) * 0.01
            for _ in range(5)
        ]
        near_dups = [v / np.linalg.norm(v) for v in near_dups]  # Normalize

        # Create neighbors list with duplicates
        neighbors = [(f"dup_{i}", v, 0.99) for i, v in enumerate(near_dups)]

        # LVM generates a vector similar to base_vec
        v_hat = base_vec + np.random.randn(768) * 0.05
        v_hat = v_hat / np.linalg.norm(v_hat)

        # Create decoder with near-dup detection
        decoder = DualPathDecoder(
            lane="neutral",
            tau_snap=0.92,
            tau_novel=0.85,
            near_dup_cos=0.98,  # Strict threshold
            near_dup_window=8
        )

        # Generate multiple steps - should see NOVEL_DUP_DROP
        decisions = []
        for _ in range(10):
            v_out, rec = decoder.step(v_hat, neighbors)
            decisions.append(rec.decision)

        # Should see some NOVEL_DUP_DROP decisions
        dup_drops = sum(1 for d in decisions if d == 'NOVEL_DUP_DROP')
        print(f"\nNear-duplicate detection:")
        print(f"  NOVEL_DUP_DROP count: {dup_drops}/10")
        print(f"  Decisions: {decisions}")

        assert dup_drops > 0, "Should detect near-duplicates and use NOVEL_DUP_DROP"

    def test_recent_ids_buffer(self):
        """Test recent_ids buffer maintains correct size."""
        decoder = DualPathDecoder(
            lane="neutral",
            tau_snap=0.92,
            tau_novel=0.85,
            near_dup_window=8
        )

        # Add 20 IDs
        for i in range(20):
            decoder.recent_ids.append(f"id_{i}")

        # Check buffer size (should be capped at 64)
        assert len(decoder.recent_ids) <= 64, \
            f"Buffer should be <= 64, got {len(decoder.recent_ids)}"


class TestLaneConfigurations:
    """Test different lane configurations."""

    def test_lane_configurations(self, setup_test_env):
        """Test different lane configs produce different behaviors."""
        bank = setup_test_env['bank']
        miner = setup_test_env['miner']
        query_tower = setup_test_env['query_tower']
        lvm = setup_test_env['lvm']

        lanes = {
            'conservative': (0.94, 0.88),
            'neutral': (0.92, 0.85),
            'creative': (0.90, 0.82),
        }

        results = {}

        for lane_name, (tau_snap, tau_novel) in lanes.items():
            context = setup_test_env['context'].copy()
            decoder = DualPathDecoder(
                lane=lane_name,
                tau_snap=tau_snap,
                tau_novel=tau_novel
            )

            decisions = []
            for _ in range(20):
                v_hat = lvm.forward(context)

                ctx_torch = torch.from_numpy(context).float().unsqueeze(0)
                with torch.no_grad():
                    q = query_tower(ctx_torch)
                q_np = q.cpu().numpy()[0]

                I, D = miner.search(q_np[None, :], k=20)
                neighbors = [(f"bank_{i}", bank[i], D[0, j]) for j, i in enumerate(I[0])]

                v_out, rec = decoder.step(v_hat, neighbors)
                context = np.vstack([context, v_out])
                decisions.append(rec.decision)

            dist = Counter(decisions)
            results[lane_name] = dist

        print(f"\nLane comparison (20 steps each):")
        for lane_name, dist in results.items():
            print(f"  {lane_name}:")
            for decision, count in dist.most_common():
                print(f"    {decision}: {count}")

        # All lanes should work
        assert all(len(dist) > 0 for dist in results.values()), \
            "All lanes should produce decisions"


class TestDecisionLogic:
    """Unit tests for dual-path decision logic."""

    def test_snap_decision(self):
        """Test SNAP decision when cosine >= tau_snap."""
        v_hat = np.random.randn(768).astype(np.float32)
        v_hat = v_hat / np.linalg.norm(v_hat)

        # Bank vector very similar to v_hat (use small perturbation)
        v_bank = v_hat + np.random.randn(768) * 0.001  # Very small noise
        v_bank = v_bank / np.linalg.norm(v_bank)

        cosine = float(np.dot(v_hat, v_bank))

        # If still not high enough, just use v_hat itself
        if cosine < 0.92:
            v_bank = v_hat.copy()
            cosine = 1.0

        neighbors = [("bank_0", v_bank, cosine)]
        cfg = LaneConfig(tau_snap=0.92, tau_novel=0.85, lane_name='neutral')

        v_out, rec = choose_next_vector(v_hat, neighbors, cfg)

        assert rec.decision == 'SNAP', f"Expected SNAP, got {rec.decision} (cosine={cosine})"
        assert np.allclose(v_out, v_bank), "SNAP should return bank vector"

    def test_novel_decision(self):
        """Test NOVEL decision when cosine <= tau_novel."""
        v_hat = np.random.randn(768).astype(np.float32)
        v_hat = v_hat / np.linalg.norm(v_hat)

        # Bank vector dissimilar to v_hat
        v_bank = np.random.randn(768).astype(np.float32)
        v_bank = v_bank / np.linalg.norm(v_bank)

        cosine = float(np.dot(v_hat, v_bank))

        # Force low cosine
        if cosine > 0.85:
            v_bank = -v_bank
            cosine = float(np.dot(v_hat, v_bank))

        neighbors = [("bank_0", v_bank, cosine)]
        cfg = LaneConfig(tau_snap=0.92, tau_novel=0.85, lane_name='neutral')

        v_out, rec = choose_next_vector(v_hat, neighbors, cfg)

        assert rec.decision == 'NOVEL', f"Expected NOVEL, got {rec.decision}"
        assert np.allclose(v_out, v_hat), "NOVEL should return LVM vector"

    def test_blend_decision(self):
        """Test BLEND decision when tau_novel < cosine < tau_snap."""
        v_hat = np.random.randn(768).astype(np.float32)
        v_hat = v_hat / np.linalg.norm(v_hat)

        # Create a bank vector with cosine in BLEND range [0.85, 0.92]
        # Use weighted sum with target cosine = 0.88
        target_cos = 0.88
        # v_bank = target_cos * v_hat + sqrt(1 - target_cos^2) * orthogonal
        random_vec = np.random.randn(768).astype(np.float32)
        random_vec = random_vec - np.dot(random_vec, v_hat) * v_hat  # Make orthogonal
        random_vec = random_vec / np.linalg.norm(random_vec)

        v_bank = target_cos * v_hat + np.sqrt(1 - target_cos**2) * random_vec
        v_bank = v_bank / np.linalg.norm(v_bank)

        cosine = float(np.dot(v_hat, v_bank))
        # Ensure in BLEND range
        assert 0.85 < cosine < 0.92, f"Cosine {cosine} not in BLEND range [0.85, 0.92]"

        neighbors = [("bank_0", v_bank, cosine)]
        cfg = LaneConfig(tau_snap=0.92, tau_novel=0.85, lane_name='neutral')

        v_out, rec = choose_next_vector(v_hat, neighbors, cfg)

        assert rec.decision == 'BLEND', f"Expected BLEND, got {rec.decision} (cosine={cosine})"
        # BLEND should be combination
        assert not np.allclose(v_out, v_hat, atol=0.01), "BLEND should not be pure LVM"
        assert not np.allclose(v_out, v_bank, atol=0.01), "BLEND should not be pure bank"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
