import numpy as np
data = np.load('artifacts/wikipedia_584k_fresh.npz')
article_indices = data['article_indices']
print(f"Min article index: {np.min(article_indices)}")
print(f"Max article index: {np.max(article_indices)}")