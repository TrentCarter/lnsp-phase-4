#!/usr/bin/env python3
"""
Compare all 5 LVM models: 80k vs 232k training data
"""

import torch
import sys
from pathlib import Path

# Model configurations
MODELS = [
    {
        'name': 'AMN',
        'path_80k': 'artifacts/lvm/models/amn_20251016_133427/best_model.pt',
        'path_232k': 'artifacts/lvm/models/amn_232k_20251017_090129/best_model.pt',
        'type': 'standard'
    },
    {
        'name': 'LSTM',
        'path_80k': 'artifacts/lvm/models/lstm_20251016_133934/best_model.pt',
        'path_232k': 'artifacts/lvm/models/lstm_232k_20251017_090129/best_model.pt',
        'type': 'standard'
    },
    {
        'name': 'GRU',
        'path_80k': 'artifacts/lvm/models/gru_20251016_134451/best_model.pt',
        'path_232k': 'artifacts/lvm/models/gru_232k_20251017_090129/best_model.pt',
        'type': 'standard'
    },
    {
        'name': 'Transformer',
        'path_80k': 'artifacts/lvm/models/transformer_20251016_135606/best_model.pt',
        'path_232k': 'artifacts/lvm/models/transformer_232k_20251017_090129/best_model.pt',
        'type': 'standard'
    },
    {
        'name': 'GraphMERT-LVM',
        'path_80k': 'artifacts/lvm/models/graphmert_lvm_80k_full/benchmark_model.pt',
        'path_232k': 'artifacts/lvm/models/graphmert_lvm_232k_20251017_090129/benchmark_model.pt',
        'type': 'graphmert'
    }
]

def extract_metrics(checkpoint_path, model_type):
    """Extract validation metrics from checkpoint"""
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        if model_type == 'graphmert':
            # GraphMERT stores history
            history = ckpt.get('history', [])
            if history:
                val_cosines = [e.get('val_cosine', 0) for e in history]
                best_val = max(val_cosines)
                best_epoch = val_cosines.index(best_val) + 1
                final_val = val_cosines[-1]
            else:
                best_val = final_val = 0
                best_epoch = 0
        else:
            # Standard models store val_cosine directly
            best_val = ckpt.get('val_cosine', 0)
            final_val = best_val
            best_epoch = ckpt.get('epoch', 0)

        # Get model size
        params = sum(p.numel() for p in ckpt['model_state_dict'].values())

        return {
            'best_val_cosine': best_val,
            'final_val_cosine': final_val,
            'best_epoch': best_epoch,
            'total_params': params
        }
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {str(e)}")
        return None

def main():
    print("=" * 80)
    print("80k vs 232k Training Comparison - All Models")
    print("=" * 80)
    print()

    results = []

    for model_config in MODELS:
        name = model_config['name']
        print(f"\n{'=' * 80}")
        print(f"Model: {name}")
        print("=" * 80)

        # Load 80k metrics
        metrics_80k = extract_metrics(model_config['path_80k'], model_config['type'])

        # Load 232k metrics
        metrics_232k = extract_metrics(model_config['path_232k'], model_config['type'])

        if not metrics_80k or not metrics_232k:
            print(f"✗ Failed to load model checkpoints")
            continue

        # Calculate improvement
        val_80k = metrics_80k['best_val_cosine']
        val_232k = metrics_232k['best_val_cosine']

        if val_80k > 0:
            improvement_pct = ((val_232k - val_80k) / val_80k) * 100
        else:
            improvement_pct = 0

        print(f"\n80k Model:")
        print(f"  Best val cosine: {val_80k:.4f}")
        print(f"  Best epoch: {metrics_80k['best_epoch']}")
        print(f"  Parameters: {metrics_80k['total_params']:,}")

        print(f"\n232k Model:")
        print(f"  Best val cosine: {val_232k:.4f}")
        print(f"  Best epoch: {metrics_232k['best_epoch']}")
        print(f"  Parameters: {metrics_232k['total_params']:,}")

        print(f"\nComparison:")
        print(f"  Dataset increase: 2.88x (80k → 232k)")
        print(f"  Val cosine change: {val_80k:.4f} → {val_232k:.4f}")
        print(f"  Improvement: {improvement_pct:+.2f}%")

        if improvement_pct > 5:
            verdict = f"✅ SIGNIFICANT IMPROVEMENT (+{improvement_pct:.2f}%)"
        elif improvement_pct > 0:
            verdict = f"✓ Modest improvement (+{improvement_pct:.2f}%)"
        elif improvement_pct > -5:
            verdict = f"≈ Similar performance ({improvement_pct:+.2f}%)"
        else:
            verdict = f"⚠️ REGRESSION ({improvement_pct:.2f}%)"

        print(f"  Verdict: {verdict}")

        results.append({
            'name': name,
            'val_80k': val_80k,
            'val_232k': val_232k,
            'improvement_pct': improvement_pct,
            'params': metrics_80k['total_params']
        })

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: 80k vs 232k Comparison")
    print("=" * 80)
    print()

    # Sort by improvement
    results.sort(key=lambda x: x['improvement_pct'], reverse=True)

    print(f"{'Rank':<6} {'Model':<18} {'80k Val':<10} {'232k Val':<10} {'Change':<12} {'Verdict'}")
    print("-" * 80)

    for idx, r in enumerate(results, 1):
        emoji = "🏆" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else ""
        verdict = "✅" if r['improvement_pct'] > 5 else "✓" if r['improvement_pct'] > 0 else "≈" if r['improvement_pct'] > -5 else "⚠️"
        print(f"{emoji:<6} {r['name']:<18} {r['val_80k']:<10.4f} {r['val_232k']:<10.4f} {r['improvement_pct']:>+10.2f}%  {verdict}")

    print()
    print("=" * 80)
    print("Key Insights:")
    print("=" * 80)

    # Find best improver
    best_improver = max(results, key=lambda x: x['improvement_pct'])
    print(f"\n🏆 Best Scaler: {best_improver['name']} (+{best_improver['improvement_pct']:.2f}%)")
    print(f"   80k: {best_improver['val_80k']:.4f} → 232k: {best_improver['val_232k']:.4f}")

    # Find worst
    worst = min(results, key=lambda x: x['improvement_pct'])
    if worst['improvement_pct'] < 0:
        print(f"\n⚠️  Regressed: {worst['name']} ({worst['improvement_pct']:.2f}%)")
        print(f"   80k: {worst['val_80k']:.4f} → 232k: {worst['val_80k']:.4f}")

    # Overall verdict
    avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
    print(f"\n📊 Average Improvement: {avg_improvement:+.2f}%")

    if avg_improvement > 5:
        print("   Verdict: ✅ Larger dataset clearly helps!")
    elif avg_improvement > 0:
        print("   Verdict: ✓ Modest benefit from more data")
    else:
        print("   Verdict: ⚠️ More data didn't help overall")

    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
