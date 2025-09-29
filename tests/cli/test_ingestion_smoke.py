import subprocess
import sys
from pathlib import Path
import pytest

# Ensure the src directory is in the Python path
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))


@pytest.mark.smoke
def test_ingest_factoid_cli_smoke():
    """Run the ingestion script on a single item to test the CLI entrypoint."""
    output_npz = Path("/tmp/smoke_test_vecs.npz")
    if output_npz.exists():
        output_npz.unlink()

    # Use the curated test TSV for the smoke test
    test_tsv = ROOT_DIR / "data" / "datasets" / "factoid-wiki" / "curated-test.tsv"

    command = [
        sys.executable, "-m", "src.ingest_factoid",
        "--file-path", str(test_tsv),
        "--file-type", "tsv",
        "--num-samples", "1",
        "--faiss-out", str(output_npz),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, f"CLI script failed with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert output_npz.exists(), "Output NPZ file was not created"
    assert "Completed ingestion of 1/1 samples" in result.stdout, "Completion message not found in stdout"

    # Clean up the created file
    output_npz.unlink()
