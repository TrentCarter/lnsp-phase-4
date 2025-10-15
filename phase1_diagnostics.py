#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.lvm.train_lstm_baseline import LSTMVectorPredictor
from app.lvm.train_helpers import sample_anchors


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=1)


def angular_dispersion(vecs: torch.Tensor) -> float:
    mean_vec = vecs.mean(dim=0)
    mean_vec = mean_vec / (mean_vec.norm() + 1e-8)
    cosines = cosine_similarity(vecs, mean_vec.unsqueeze(0).expand_as(vecs))
    return float(1.0 - cosines.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='artifacts/lvm/evaluation/phase1_diagnostics.json')
    parser.add_argument('--samples', type=int, default=512)
    args = parser.parse_args()

    data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
    contexts = torch.tensor(data['context_sequences'][26385:], dtype=torch.float32)
    targets = torch.tensor(data['target_vectors'][26385:], dtype=torch.float32)

    model = LSTMVectorPredictor()
    ckpt = torch.load(args.model, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    samples = min(args.samples, contexts.size(0))
    with torch.no_grad():
        pred_raw, pred_cos = model(contexts[:samples], return_raw=True)

    anchors, sigma = sample_anchors(data['target_vectors'], 1024)
    anchors = anchors.to(pred_cos.device)

    nn_cosines = []
    for vec in torch.nn.functional.normalize(pred_cos, dim=1):
        sims = torch.matmul(anchors, vec)
        nn_cosines.append(float(sims.max().item()))

    cat = torch.nn.functional.normalize(torch.cat([anchors, pred_cos], dim=0), dim=1)
    disp_preds = angular_dispersion(torch.nn.functional.normalize(pred_cos, dim=1))
    disp_targets = angular_dispersion(torch.nn.functional.normalize(targets[:samples], dim=1))

    stats = {
        'model': args.model,
        'samples': samples,
        'nearest_cos_mean': float(np.mean(nn_cosines)),
        'nearest_cos_std': float(np.std(nn_cosines)),
        'dispersion_preds': disp_preds,
        'dispersion_targets': disp_targets,
        'sigma': sigma,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
