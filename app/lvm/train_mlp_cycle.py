#!/usr/bin/env python3
"""
Phase 3: MLP with Light Cycle Consistency

Adds cycle loss to 20-30% of training batches:
  cycle_loss = lambda_cycle * (1 - cos(pred, encode(decode(pred))))

This teaches the model to stay on vec2text's decodable manifold.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import requests
import random


class SimpleMLP(nn.Module):
    """Very simple MLP: flatten context → hidden → output"""

    def __init__(self, context_size=5, vector_dim=768, hidden_dim=1024):
        super().__init__()

        input_dim = context_size * vector_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vector_dim)
        )

        # Normalize output to unit sphere
        self.normalize = True

    def forward(self, x):
        # x shape: (batch_size, context_size, vector_dim)
        batch_size = x.shape[0]

        # Flatten context
        x_flat = x.view(batch_size, -1)

        # Pass through network
        out = self.network(x_flat)

        # Normalize to unit sphere
        if self.normalize:
            out = out / (torch.norm(out, dim=1, keepdim=True) + 1e-8)

        return out


class SequenceDataset(Dataset):
    """Dataset for sequence prediction"""

    def __init__(self, contexts, targets):
        self.contexts = torch.from_numpy(contexts).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]


def cosine_similarity_loss(predictions, targets):
    """1 - cosine similarity (minimize)"""
    # Normalize
    pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
    target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)

    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=1)

    # Loss: 1 - cosine (minimize)
    loss = 1.0 - cos_sim
    return loss.mean()


def vec2text_encode(texts, encoder_url="http://localhost:8767/embed"):
    """Encode texts to 768D vectors using vec2text-compatible encoder"""
    try:
        response = requests.post(
            encoder_url,
            json={"texts": texts},
            timeout=10
        )
        if response.status_code == 200:
            embeddings = response.json()["embeddings"]
            return torch.tensor(embeddings, dtype=torch.float32)
        return None
    except Exception as e:
        print(f"Encoding error: {e}")
        return None


def vec2text_decode(vectors, decoder_url="http://localhost:8766/decode"):
    """Decode 768D vectors to text using vec2text (now optimized!)"""
    try:
        # Convert to list
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.detach().cpu().numpy().tolist()

        response = requests.post(
            decoder_url,
            json={
                "vectors": vectors,
                "subscribers": "jxe",  # Use JXE only for speed
                "steps": 1,
                "device": "cpu"
            },
            timeout=30  # Reduced from previous ~10s per vector
        )

        if response.status_code == 200:
            result = response.json()
            # Extract decoded texts
            decoded_texts = []
            for item in result["results"]:
                jxe_output = item["subscribers"].get("gtr → jxe", {})
                decoded_texts.append(jxe_output.get("output", ""))
            return decoded_texts
        return None
    except Exception as e:
        print(f"Decoding error: {e}")
        return None


def compute_cycle_loss(predictions, encoder_url, decoder_url):
    """
    Compute cycle consistency loss: 1 - cos(pred, encode(decode(pred)))

    With the new in-memory vec2text server, this is now 10-15x faster!
    """
    batch_size = predictions.shape[0]

    # Decode predictions to text
    decoded_texts = vec2text_decode(predictions, decoder_url)
    if decoded_texts is None:
        return None

    # Re-encode texts back to vectors
    reencoded = vec2text_encode(decoded_texts, encoder_url)
    if reencoded is None:
        return None

    # Move to same device as predictions
    reencoded = reencoded.to(predictions.device)

    # Normalize both
    pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
    reenc_norm = reencoded / (torch.norm(reencoded, dim=1, keepdim=True) + 1e-8)

    # Cycle loss: 1 - cosine
    cycle_cos = (pred_norm * reenc_norm).sum(dim=1)
    cycle_loss = 1.0 - cycle_cos.mean()

    return cycle_loss


def train_epoch(model, dataloader, optimizer, device, args, epoch):
    """Train for one epoch with optional cycle consistency"""
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_cycle_loss = 0.0
    total_cosine = 0.0
    num_cycle_batches = 0

    for batch_idx, (contexts, targets) in enumerate(dataloader):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        predictions = model(contexts)

        # Main loss (always)
        main_loss = cosine_similarity_loss(predictions, targets)

        # Decide if this batch gets cycle loss
        use_cycle = random.random() < args.cycle_freq

        if use_cycle and (batch_idx % 5 == 0):  # Limit frequency to avoid slowdown
            cycle_loss = compute_cycle_loss(
                predictions,
                args.encoder_url,
                args.decoder_url
            )

            if cycle_loss is not None:
                total_loss_batch = main_loss + args.lambda_cycle * cycle_loss
                total_cycle_loss += cycle_loss.item()
                num_cycle_batches += 1
            else:
                total_loss_batch = main_loss
        else:
            total_loss_batch = main_loss

        # Backward
        total_loss_batch.backward()
        optimizer.step()

        # Track metrics
        total_loss += total_loss_batch.item()
        total_main_loss += main_loss.item()

        # Compute cosine for monitoring
        with torch.no_grad():
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_main_loss = total_main_loss / len(dataloader)
    avg_cycle_loss = total_cycle_loss / num_cycle_batches if num_cycle_batches > 0 else 0.0
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_main_loss, avg_cycle_loss, avg_cosine, num_cycle_batches


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0

    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(contexts)

            # Loss
            loss = cosine_similarity_loss(predictions, targets)

            # Cosine
            pred_norm = predictions / (torch.norm(predictions, dim=1, keepdim=True) + 1e-8)
            target_norm = targets / (torch.norm(targets, dim=1, keepdim=True) + 1e-8)
            cosine = (pred_norm * target_norm).sum(dim=1).mean()

            total_loss += loss.item()
            total_cosine += cosine.item()

    avg_loss = total_loss / len(dataloader)
    avg_cosine = total_cosine / len(dataloader)

    return avg_loss, avg_cosine


def main():
    parser = argparse.ArgumentParser(description="Train MLP with cycle consistency")
    parser.add_argument("--data", type=str, required=True, help="Path to training_sequences_ctx5.npz")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (reduced for cycle loss)")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--lambda-cycle", type=float, default=0.2, help="Cycle loss weight")
    parser.add_argument("--cycle-freq", type=float, default=0.25, help="Fraction of batches with cycle loss (0.2-0.3)")
    parser.add_argument("--encoder-url", type=str, default="http://localhost:8767/embed", help="Encoder API URL")
    parser.add_argument("--decoder-url", type=str, default="http://localhost:8766/decode", help="Decoder API URL")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model (optional)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API availability
    print("Checking vec2text APIs...")
    try:
        enc_resp = requests.get(args.encoder_url.replace("/embed", "/health"), timeout=5)
        if enc_resp.status_code == 200:
            print(f"✓ Encoder API available (optimized in-memory)")
        else:
            print(f"✗ Encoder API not available")
            return
    except:
        print(f"✗ Could not connect to encoder API")
        return

    try:
        dec_resp = requests.get(args.decoder_url.replace("/decode", "/health"), timeout=5)
        if dec_resp.status_code == 200:
            dec_info = dec_resp.json()
            print(f"✓ Decoder API available (mode: {dec_info.get('mode', 'unknown')})")
        else:
            print(f"✗ Decoder API not available")
            return
    except:
        print(f"✗ Could not connect to decoder API")
        return

    print()

    # Load data
    print(f"Loading data from {args.data}")
    data = np.load(args.data)
    contexts = data['context_sequences']
    targets = data['target_vectors']

    # Train/val split (90/10)
    num_samples = len(contexts)
    train_size = int(0.9 * num_samples)

    train_contexts = contexts[:train_size]
    train_targets = targets[:train_size]
    val_contexts = contexts[train_size:]
    val_targets = targets[train_size:]

    print(f"Training samples: {len(train_contexts)}")
    print(f"Validation samples: {len(val_contexts)}")

    # Create datasets
    train_dataset = SequenceDataset(train_contexts, train_targets)
    val_dataset = SequenceDataset(val_contexts, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    device = torch.device(args.device)
    model = SimpleMLP(
        context_size=5,
        vector_dim=768,
        hidden_dim=args.hidden_dim
    ).to(device)

    # Load pretrained if specified
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Pretrained model loaded")

    print(f"\nModel architecture:")
    print(f"  Input: 5x768D (flattened to 3840D)")
    print(f"  Hidden: {args.hidden_dim}D")
    print(f"  Output: 768D (normalized)")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"Cycle consistency settings:")
    print(f"  Lambda (weight): {args.lambda_cycle}")
    print(f"  Frequency: {args.cycle_freq * 100:.0f}% of batches")
    print(f"  Expected batches/epoch: ~{int(len(train_loader) * args.cycle_freq)}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        train_loss, train_main, train_cycle, train_cosine, num_cycle = train_epoch(
            model, train_loader, optimizer, device, args, epoch
        )
        val_loss, val_cosine = validate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f} (main: {train_main:.4f}, cycle: {train_cycle:.4f}, batches: {num_cycle}) | "
              f"Train Cosine: {train_cosine:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Cosine: {val_cosine:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_main_loss': float(train_main),
            'train_cycle_loss': float(train_cycle),
            'train_cosine': float(train_cosine),
            'val_loss': float(val_loss),
            'val_cosine': float(val_cosine),
            'num_cycle_batches': num_cycle
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cosine': val_cosine,
                'args': vars(args)
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_cosine': val_cosine,
        'args': vars(args)
    }, output_dir / 'final_model.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
