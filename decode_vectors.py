
import numpy as np
import requests
import json
from tqdm import tqdm

# Load the target vectors
data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
target_vectors = data['target_vectors']

# Decode the first 100 vectors
decoded_texts = []
for i in tqdm(range(100)):
    vector = target_vectors[i].tolist()
    try:
        response = requests.post("http://localhost:8767/decode", json={"vectors": [vector]}, timeout=60)
        if response.status_code == 200:
            decoded_text = response.json()['results'][0]['output']
            if not decoded_text:
                print(f"Empty text for vector {i}")
            decoded_texts.append(decoded_text)
        else:
            print(f"Error for vector {i}: {response.status_code} {response.text}")
            decoded_texts.append("")
    except requests.exceptions.Timeout:
        print(f"Timeout for vector {i}")
        decoded_texts.append("")

# Save the decoded texts to a file
with open('decoded_texts.jsonl', 'w') as f:
    for text in decoded_texts:
        f.write(json.dumps({"text": text}) + '\n')
