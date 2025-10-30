import numpy as np
import torch
import torch.nn as nn
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
from app.lvm.models import create_model
import os

def get_model_and_data():
    """
    Initializes and returns the GRU model, the text orchestrator, and sample data.
    """
    # Load the orchestrator for encoding text
    orchestrator = IsolatedVecTextVectOrchestrator()

    # Sample text for testing
    sample_text = ["This is a test sentence.", "Another sentence for testing purposes.", "The quick brown fox jumps over the lazy dog.", "Never underestimate the power of a good book.", "The journey of a thousand miles begins with a single step.", "In the middle of difficulty lies opportunity.", "That which does not kill us makes us stronger.", "The only true wisdom is in knowing you know nothing."]
    
    # Encode the text to get vectors
    vectors = orchestrator.encode_texts(sample_text)

    # Create sequences of length 5
    context_len = 5
    sequences = []
    for i in range(len(vectors) - context_len + 1):
        sequence = vectors[i:i+context_len]
        sequences.append(sequence.cpu().numpy())
    data = np.array(sequences)
    
    # Initialize the GRU model
    model = create_model('gru')
    
    # Load the trained model checkpoint
    model_path = "artifacts/lvm/models/gru_20251016_134451/best_model.pt"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using an untrained model.")

    model.eval()
    
    return model, orchestrator, torch.tensor(data, dtype=torch.float32)

def print_stats(name, tensor):
    """Calculates and prints the mean and standard deviation of a tensor."""
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    print(f"--- {name} ---")
    print(f"  Mean (overall): {tensor.mean():.6f}")
    print(f"  Std (overall): {tensor.std():.6f}")
    print(f"  Mean (per-dim, first 5): {mean[:5].tolist()}")
    print(f"  Std (per-dim, first 5): {std[:5].tolist()}")
    print("\n")

def main():
    """
    Main function to run the moment matching test.
    """
    model, orchestrator, data = get_model_and_data()

    # --- GTR-T5 Encoder Output ---
    encoder_output = data
    print_stats("GTR-T5 Encoder Output", encoder_output)

    # --- LVM Model Output (Pre-Normalization) ---
    # Create a dummy context
    context = data

    # Move data to the same device as the model
    device = next(model.parameters()).device
    context = context.to(device)

    print(f"Context shape: {context.shape}")
    
    with torch.no_grad():
        model_output_pre_norm, model_output_post_norm = model(context, return_raw=True)

    print_stats("LVM Model Output (Pre-Normalization)", model_output_pre_norm)

    # --- LVM Model Output (Post-Normalization) ---
    print_stats("LVM Model Output (Post-Normalization)", model_output_post_norm)

    print("--- Expected Behavior ---")
    print("GTR-T5 Encoder: Mean should be close to 0, with varying std per dimension.")
    print("LVM Pre-Norm: Should ideally have a similar distribution to the GTR-T5 encoder output.")
    print("LVM Post-Norm: Std should be constant and low, indicating the collapsed variance problem.")

if __name__ == "__main__":
    main()
