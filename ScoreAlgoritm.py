import numpy as np

n_objects = 90
n_features = 21
base_set = np.random.rand(n_objects, n_features)

Abolfazl 🤘, [11 / 20 / 2025 11: 25
AM]

def calculate_prox_vector(base_set):
    n = len(base_set)
    e = np.zeros(n_features)
    for j in range(n_features):
        x = 0
        for i in range(n):
            x += np.abs(base_set[i, j - 1] - base_set[i, j])
        e[j] = x

    return e


calculate_prox_vector(base_set)