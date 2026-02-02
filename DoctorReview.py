import numpy as np
import time

# تنظیم Seed برای تکرارپذیری نتایج
np.random.seed(42)

# ==========================================
# 1. تولید داده (همانند قبل)
# ==========================================
n_patients = 90
n_features = 21

X = np.column_stack([
    np.random.normal(120, 20, n_patients),  # Systolic BP
    np.random.normal(80, 10, n_patients),  # Diastolic BP
    np.random.normal(200, 40, n_patients),  # Cholesterol
    np.random.normal(5.5, 1.0, n_patients),  # FBS
    np.random.normal(25, 5, n_patients),  # BMI
    np.random.normal(4.5, 0.5, n_patients),  # Hemoglobin
    np.random.normal(7.0, 1.5, n_patients),  # Cr
    np.random.normal(140, 10, n_patients),  # Na
    np.random.normal(4.0, 0.5, n_patients),  # K
    np.random.normal(250, 50, n_patients),  # TG (Large numbers!)
    np.random.normal(50, 10, n_patients),  # HDL
    np.random.normal(150, 30, n_patients),  # LDL
    np.random.normal(3.0, 0.8, n_patients),  # Troponin
    np.random.normal(60, 15, n_patients),  # HR
    np.random.normal(98, 2, n_patients),  # O2 Sat
    np.random.normal(40, 5, n_patients),  # HCT
    np.random.normal(9.5, 1.0, n_patients),  # WBC
    np.random.normal(250, 50, n_patients),  # PLT (Very large!)
    np.random.normal(10.0, 2.0, n_patients),  # AST
    np.random.normal(8.0, 1.5, n_patients),  # ALT
    np.random.normal(30, 10, n_patients),  # CPK
])

query = np.array([
    134.0, 85.0, 210.0, 5.7, 27.0,
    4.8, 7.2, 138.0, 4.1, 260.0,
    48.0, 155.0, 3.2, 62.0, 97.0,
    39.0, 10.0, 240.0, 11.0, 8.5, 35.0
])


# ==========================================
# 2. الگوریتم بهینه‌شده (Refactored)
# ==========================================
def improved_scoring_algorithm(X, query, k=10, n_iterations=5000, L=5):
    """
    این تابع از نرمال‌سازی و نمونه‌برداری تصادفی (Monte Carlo) استفاده می‌کند.
    """
    n_patients, n_features = X.shape

    # --- گام ۱: نرمال‌سازی داده‌ها (Z-Score) ---
    # فرمول: (x - mean) / std
    # این کار باعث می‌شود همه ویژگی‌ها وزن یکسانی داشته باشند
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # جلوگیری از تقسیم بر صفر (اگر انحراف معیار صفر باشد)
    std[std == 0] = 1

    X_scaled = (X - mean) / std
    query_scaled = (query - mean) / std

    print(f"[Improved] Running Monte Carlo simulation with {n_iterations:,} iterations...")

    votes = np.zeros(n_patients, dtype=int)

    # --- گام ۲: رأی‌گیری با نمونه‌برداری تصادفی ---
    for _ in range(n_iterations):
        # انتخاب تصادفی k ویژگی (Subspace)
        # replace=False یعنی ویژگی تکراری انتخاب نشود
        subset_indices = np.random.choice(n_features, k, replace=False)

        # محاسبه فاصله اقلیدسی فقط روی ویژگی‌های انتخاب شده
        # (استفاده از توان ۲ به جای رادیکال برای سرعت بیشتر، چون ترتیب تغییر نمی‌کند)
        dist_sq = np.sum((X_scaled[:, subset_indices] - query_scaled[subset_indices]) ** 2, axis=1)

        # پیدا کردن ایندکسِ کمترین فاصله
        closest_patient_idx = np.argmin(dist_sq)
        votes[closest_patient_idx] += 1

    # انتخاب L نفر برتر
    top_L_indices = np.argsort(-votes)[:L]
    return top_L_indices, votes


# ==========================================
# 3. اجرا و مقایسه
# ==========================================
start = time.time()

# اجرا با ۵۰۰۰ تکرار (بسیار کمتر از ۳۵۰،۰۰۰ حالت اصلی، اما آماری دقیق)
top_new, votes_new = improved_scoring_algorithm(X, query, k=10, n_iterations=5000, L=5)

end = time.time()

print("\n--- Results ---")
print("Top similar patients (indices):", top_new)
print("Votes received:", votes_new[top_new])
print(f"Total Time: {end - start:.4f} seconds")