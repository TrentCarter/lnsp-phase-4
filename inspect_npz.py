import numpy as np
data = np.load('artifacts/lvm/training_sequences_ctx5.npz')
for key in data.keys():
    print(key)