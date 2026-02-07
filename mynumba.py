import numpy as np
import time
import math
import itertools
from numba import njit

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


# --- 2. NUMBA CORE LOGIC ---
@njit
def numba_core_scoring_serial(X, query, subsets):
    """
    این تابع توسط کامپایلر Numba به کد ماشین تبدیل می‌شود.
    حلقه‌ها در اینجا بسیار سریع اجرا می‌شوند.
    """
    n_patients = X.shape[0]
    n_subsets = subsets.shape[0]
    votes = np.zeros(n_patients, dtype=np.int32)

    if n_subsets == 0:
        return votes

    k = subsets.shape[1]

    for i in range(n_subsets):
        subset = subsets[i]

        # محاسبه فاصله اولین بیمار برای شروع مقایسه
        min_dist = 0.0
        for j in range(k):
            feat_idx = subset[j]
            min_dist += np.abs(X[0, feat_idx] - query[feat_idx])
        closest = 0

        # بررسی بقیه بیماران
        for p in range(1, n_patients):
            current_dist = 0.0
            # حلقه دستی برای سرعت بیشتر در Numba
            for j in range(k):
                feat_idx = subset[j]
                current_dist += np.abs(X[p, feat_idx] - query[feat_idx])

            if current_dist < min_dist:
                min_dist = current_dist
                closest = p

        votes[closest] += 1

    return votes


def numba_scoring_wrapper(X, query, k=None, L=5):
    n, m = X.shape
    if k is None:
        k = m // 2

    total_comb = math.comb(m, k)
    print(f"[Numba] Processing {total_comb:,} subsets...")

    # تبدیل ترکیب‌ها به آرایه نامپای (چون Numba با لیست پایتون کند است)
    subsets_list = list(itertools.combinations(range(m), k))
    subsets_np = np.array(subsets_list, dtype=np.int32)

    # اجرای تابع کامپایل شده
    votes = numba_core_scoring_serial(X, query, subsets_np)

    top_L = np.argsort(-votes)[:L]
    return top_L, votes


# --- 3. EXECUTION ---
if __name__ == "__main__":
    k_val = 10
    L_val = 5

    print("=" * 60)
    print("Running NUMBA Algorithm...")

    # Warm-up (کامپایل اولیه)
    print("Compiling (Warm-up)...")
    _ = numba_scoring_wrapper(X, query, k=k_val, L=L_val)

    # اجرای اصلی و زمان‌گیری
    start_time = time.time()
    top_patients, vote_counts = numba_scoring_wrapper(X, query, k=k_val, L=L_val)
    end_time = time.time()

    print(f"Top L={L_val} Patients: {top_patients}")
    print(f"Votes: {vote_counts[top_patients]}")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print("=" * 60)