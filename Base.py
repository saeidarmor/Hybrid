import numpy as np
import time
import itertools

# --- 1. DATA GENERATION ---
np.random.seed(42)
n_patients = 90
n_features = 21

X = np.column_stack([
    np.random.normal(120, 20, n_patients),
    np.random.normal(80, 10, n_patients),
    np.random.normal(200, 40, n_patients),
    np.random.normal(5.5, 1.0, n_patients),
    np.random.normal(25, 5, n_patients),
    np.random.normal(4.5, 0.5, n_patients),
    np.random.normal(7.0, 1.5, n_patients),
    np.random.normal(140, 10, n_patients),
    np.random.normal(4.0, 0.5, n_patients),
    np.random.normal(250, 50, n_patients),
    np.random.normal(50, 10, n_patients),
    np.random.normal(150, 30, n_patients),
    np.random.normal(3.0, 0.8, n_patients),
    np.random.normal(60, 15, n_patients),
    np.random.normal(98, 2, n_patients),
    np.random.normal(40, 5, n_patients),
    np.random.normal(9.5, 1.0, n_patients),
    np.random.normal(250, 50, n_patients),
    np.random.normal(10.0, 2.0, n_patients),
    np.random.normal(8.0, 1.5, n_patients),
    np.random.normal(30, 10, n_patients),
])

query = np.array([
    134.0, 85.0, 210.0, 5.7, 27.0, 4.8, 7.2, 138.0, 4.1, 260.0,
    48.0, 155.0, 3.2, 62.0, 97.0, 39.0, 10.0, 240.0, 11.0, 8.5, 35.0
])


# --- 2. ORIGINAL ALGORITHM ---
def original_scoring_algorithm(X, query, k=None, L=5):
    n, m = X.shape
    if k is None:
        k = m // 2

    votes = np.zeros(n, dtype=int)

    # Generate all combinations
    subsets = itertools.combinations(range(m), k)

    # Iterate through subsets (This is the SLOW part in Python)
    for subset in subsets:
        subset_idx = list(subset)

        # Slicing for current subset
        sub_X = X[:, subset_idx]
        sub_q = query[subset_idx]

        # Manhattan distance (L1) calculation using NumPy vectorization
        dists = np.sum(np.abs(sub_X - sub_q), axis=1)

        # Find closest patient
        closest_patient = np.argmin(dists)
        votes[closest_patient] += 1

    top_L = np.argsort(-votes)[:L]
    return top_L, votes


# --- 3. EXECUTION ---
if __name__ == "__main__":
    k_val = 10
    L_val = 5

    print("=" * 60)
    print("Running ORIGINAL Algorithm...")
    start_time = time.time()

    top_patients, vote_counts = original_scoring_algorithm(X, query, k=k_val, L=L_val)

    end_time = time.time()

    print(f"Top L={L_val} Patients: {top_patients}")
    print(f"Votes: {vote_counts[top_patients]}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print("=" * 60)