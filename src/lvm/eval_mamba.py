"""
Evaluate trained LVM on test set.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.lvm.models import LatentMamba
from src.lvm.train_mamba import load_training_data, evaluate


def test_lvm(
    model_path: str = "models/lvm_wordnet.pt",
    data_path: str = "artifacts/lvm/wordnet_training_sequences.npz",
    device: str = "mps"
):
    """Evaluate LVM on test set."""
    print("=== Evaluating LVM ===")

    # Load data
    data = load_training_data(data_path)
    test_ctx, test_tgt, test_mask = data['test']

    print(f"Test: {len(test_ctx)} sequences")

    # Create dataloader
    test_dataset = TensorDataset(
        torch.from_numpy(test_ctx).float(),
        torch.from_numpy(test_tgt).float(),
        torch.from_numpy(test_mask).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Load model
    d_input = test_ctx.shape[2]
    model = LatentMamba(d_input=d_input, d_hidden=512, n_layers=2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Evaluate
    test_loss = evaluate(model, test_loader, device)

    print(f"Test MSE loss: {test_loss:.6f}")
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    test_lvm()
