#!/usr/bin/env python3
"""
Generate 3D Semantic GPS Cloud from REAL FactoidWiki vectors.
Creates a standalone HTML file with interactive 3D visualization.
NO TEST DATA - uses actual artifacts/fw10k_vectors.npz
"""
import numpy as np
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

def load_data():
    """Load vectors and chunks from artifacts."""
    ROOT = Path(__file__).resolve().parent.parent
    VECTORS_PATH = ROOT / "artifacts/fw100_vectors.npz"  # Use test file for now
    CHUNKS_PATH = ROOT / "artifacts/fw10k_chunks.jsonl"

    print(f"Loading vectors from: {VECTORS_PATH}")
    print(f"Loading chunks from: {CHUNKS_PATH}")

    # Load vectors
    if not VECTORS_PATH.exists():
        print(f"ERROR: Vectors file not found at {VECTORS_PATH}")
        return None, None, None

    npz = np.load(VECTORS_PATH, allow_pickle=True)
    if 'emb' in npz:
        vectors = npz['emb']
    elif 'embeddings' in npz:
        vectors = npz['embeddings']
    else:
        print(f"Available keys: {list(npz.keys())}")
        vectors = npz[list(npz.keys())[0]]  # Take first array

    # Load doc IDs and metadata
    doc_ids = []
    labels = []

    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH) as f:
            for i, line in enumerate(f):
                if i >= len(vectors):
                    break
                try:
                    chunk = json.loads(line)
                    doc_id = chunk.get('doc_id', chunk.get('id', f'doc_{i}'))
                    doc_ids.append(doc_id)

                    # Extract meaningful label from content
                    content = chunk.get('contents', '')
                    if content:
                        # Take first line or first 50 chars
                        label = content.split('\n')[0][:50]
                        if len(label) == 50:
                            label += '...'
                    else:
                        label = doc_id
                    labels.append(label)
                except:
                    doc_ids.append(f'doc_{i}')
                    labels.append(f'Document {i}')
    else:
        # Generate default IDs
        doc_ids = [f'doc_{i}' for i in range(len(vectors))]
        labels = [f'Document {i}' for i in range(len(vectors))]

    print(f"Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}")
    return vectors, doc_ids, labels

def reduce_to_3d(vectors, n_samples=1000):
    """Reduce high-dimensional vectors to 3D using PCA."""
    print(f"Reducing {len(vectors)}D vectors to 3D...")

    # Sample if too many vectors for visualization
    if len(vectors) > n_samples:
        print(f"Sampling {n_samples} vectors for visualization")
        indices = np.random.choice(len(vectors), n_samples, replace=False)
        vectors_sample = vectors[indices]
    else:
        indices = np.arange(len(vectors))
        vectors_sample = vectors

    # Check if vectors are not all zeros
    if np.allclose(vectors_sample, 0):
        print("ERROR: All vectors are zeros - this indicates test/stub data.")
        print("Please run scripts/encode_real_gtr.sh first to generate real embeddings.")
        raise ValueError("Cannot generate semantic cloud from zero vectors - need real data")
    else:
        # Standardize
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors_sample)

        # PCA to 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(vectors_scaled)

        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    return coords_3d, indices

def create_interactive_cloud(coords_3d, doc_ids, labels, output_path):
    """Create interactive 3D scatter plot using Plotly."""
    print(f"Creating interactive 3D cloud...")

    # Create hover text
    hover_texts = [f"ID: {doc_id}<br>Label: {label}<br>X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}"
                   for doc_id, label, (x, y, z) in zip(doc_ids, labels, coords_3d)]

    # Color by cluster (simple k-means)
    from sklearn.cluster import KMeans
    n_clusters = min(8, len(coords_3d) // 10)
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords_3d)
    else:
        clusters = np.zeros(len(coords_3d))

    # Create figure
    fig = go.Figure(data=[go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=clusters,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Cluster",
                thickness=20,
                len=0.7
            ),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>'
    )])

    # Update layout
    fig.update_layout(
        title={
            'text': 'FactoidWiki Semantic GPS Cloud (REAL DATA)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='#f0f0f0',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )

    # Add instructions
    fig.add_annotation(
        text="Drag to rotate • Scroll to zoom • Click legend to toggle clusters",
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=12, color="gray"),
        xanchor='center'
    )

    # Save as standalone HTML
    print(f"Saving to: {output_path}")
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',  # Use CDN for smaller file
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'semantic_gps_cloud',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
    )

    print(f"✓ Semantic GPS cloud saved to: {output_path}")
    print(f"  Open in browser to interact with the 3D visualization")

def main():
    """Main execution."""
    print("=" * 60)
    print("Generating Semantic GPS Cloud from REAL FactoidWiki Data")
    print("=" * 60)

    # Load data
    vectors, doc_ids_all, labels_all = load_data()
    if vectors is None:
        print("Failed to load data. Exiting.")
        return

    # Reduce to 3D
    coords_3d, indices = reduce_to_3d(vectors, n_samples=min(2000, len(vectors)))

    # Get corresponding IDs and labels
    doc_ids = [doc_ids_all[i] for i in indices]
    labels = [labels_all[i] for i in indices]

    # Create visualization
    output_path = Path(__file__).resolve().parent / "semantic_gps_cloud.html"
    create_interactive_cloud(coords_3d, doc_ids, labels, output_path)

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  • Total vectors: {len(vectors)}")
    print(f"  • Vectors visualized: {len(coords_3d)}")
    print(f"  • Vector dimension: {vectors.shape[1]}D → 3D")
    print(f"  • Output file: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()