import numpy as np
import argparse

def cosine_similarity(v1, v2):
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def analyze_data(file_path, num_samples=100):
    """
    Analyzes the forward-distinctness of the dataset.
    """
    try:
        data = np.load(file_path)
        contexts = data['context_sequences']
        targets = data['target_vectors']
        print(f"✅ Successfully loaded data from {file_path}")
        print(f"   - Contexts shape: {contexts.shape}")
        print(f"   - Targets shape: {targets.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    num_total_samples = contexts.shape[0]
    if num_total_samples < num_samples:
        num_samples = num_total_samples

    # Get random indices for sampling
    random_indices = np.random.choice(num_total_samples, size=num_samples, replace=False)

    sim_last_list = []
    sim_second_last_list = []
    delta_list = []

    for i in random_indices:
        target_vec = targets[i]
        ctx_last_vec = contexts[i, -1, :]
        ctx_second_last_vec = contexts[i, -2, :] if contexts.shape[1] > 1 else None

        sim_last = cosine_similarity(target_vec, ctx_last_vec)
        sim_last_list.append(sim_last)

        delta = 1.0 - sim_last
        delta_list.append(delta)

        if ctx_second_last_vec is not None:
            sim_second_last = cosine_similarity(target_vec, ctx_second_last_vec)
            sim_second_last_list.append(sim_second_last)

    avg_sim_last = np.mean(sim_last_list)
    avg_delta = np.mean(delta_list)
    avg_sim_second_last = np.mean(sim_second_last_list) if sim_second_last_list else "N/A"

    print("\n--- Analysis Results ---")
    print(f"Ran on {num_samples} random samples.")
    print(f"Average cos(target, ctx[-1]):      {avg_sim_last:.4f}")
    if avg_sim_second_last != "N/A":
        print(f"Average cos(target, ctx[-2]):      {avg_sim_second_last:.4f}")
    print(f"Average delta (1 - cos(target, ctx[-1])): {avg_delta:.4f}")

    print("\n--- Sample Spot Check (first 5 samples) ---")
    for i, idx in enumerate(random_indices[:5]):
        sim_last = sim_last_list[i]
        sim_second_last = sim_second_last_list[i] if sim_second_last_list else "N/A"
        print(f"Sample {i+1}: cos(target, ctx[-1])={sim_last:.4f}", end="")
        if sim_second_last != "N/A":
            print(f", cos(target, ctx[-2])={sim_second_last:.4f}")
        else:
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify forward-distinctness of LVM curriculum data.")
    parser.add_argument(
        "--file-path",
        type=str,
        default="artifacts/lvm/training_sequences_ctx5_584k_clean_splits_stage_a_top30.npz",
        help="Path to the .npz dataset file."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples to analyze."
    )
    args = parser.parse_args()
    analyze_data(args.file_path, args.num_samples)
