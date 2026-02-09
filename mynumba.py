import numpy as np
import time
import math
import itertools
import numba
from numba import njit, prange


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


# --- 2. PARALLEL NUMBA CORE ---
@njit(parallel=True)
def numba_core_scoring_parallel(X, query, subsets):
    n_patients = X.shape[0]
    n_subsets = subsets.shape[0]
    k = subsets.shape[1]

    n_threads = numba.get_num_threads()
    local_votes = np.zeros((n_threads, n_patients), dtype=np.int32)

    for i in prange(n_subsets):
        tid = numba.get_thread_id()
        subset = subsets[i]

        # distance for patient 0
        min_dist = 0.0
        for j in range(k):
            idx = subset[j]
            diff = X[0, idx] - query[idx]
            if diff < 0:
                diff = -diff
            min_dist += diff
        closest = 0

        # remaining patients
        for p in range(1, n_patients):
            dist = 0.0
            for j in range(k):
                idx = subset[j]
                diff = X[p, idx] - query[idx]
                if diff < 0:
                    diff = -diff
                dist += diff

            if dist < min_dist:
                min_dist = dist
                closest = p

        local_votes[tid, closest] += 1

    # reduction
    votes = np.zeros(n_patients, dtype=np.int32)
    for t in range(n_threads):
        for p in range(n_patients):
            votes[p] += local_votes[t, p]

    return votes


def numba_scoring_wrapper(X, query, k=None, L=5):
    n, m = X.shape
    if k is None:
        k = m // 2

    total_comb = math.comb(m, k)
    print(f"[Numba] Processing {total_comb:,} subsets")

    subsets_np = np.array(
        list(itertools.combinations(range(m), k)),
        dtype=np.int32
    )

    votes = numba_core_scoring_parallel(X, query, subsets_np)
    top_L = np.argsort(-votes)[:L]

    return top_L, votes


# --- 3. EXECUTION ---
if __name__ == "__main__":
    k_val = 10
    L_val = 5

    print("=" * 60)
    print("Running PARALLEL NUMBA Algorithm")

    print("Compiling (warm-up)...")
    _ = numba_scoring_wrapper(X, query, k=k_val, L=L_val)

    start = time.time()
    top_patients, vote_counts = numba_scoring_wrapper(X, query, k=k_val, L=L_val)
    end = time.time()

    print(f"Top L={L_val} Patients:", top_patients)
    print("Votes:", vote_counts[top_patients])
    print(f"Time Taken: {end - start:.4f} seconds")
    print("=" * 60)
