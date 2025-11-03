
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the keys of a .npz file.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the .npz file to inspect."
    )
    args = parser.parse_args()

    try:
        data = np.load(args.file_path)
        print(f"✅ Keys in {args.file_path}:")
        for key in data.keys():
            print(f"   - {key}")
    except Exception as e:
        print(f"❌ Error loading or inspecting file: {e}")
