import numpy as np
import pandas as pd
import time

np.random.seed(42)

# ==========================================
# 1. تعریف نام ویژگی‌ها و تولید داده
# ==========================================
feature_names = [
    "Systolic BP", "Diastolic BP", "Cholesterol", "Fasting Sugar", "BMI",
    "Hemoglobin", "Creatinine", "Sodium (Na)", "Potassium (K)", "Triglycerides",
    "HDL", "LDL", "Troponin", "Heart Rate", "O2 Saturation",
    "HCT", "WBC", "Platelets (PLT)", "AST", "ALT", "CPK"
]

n_patients = 90
n_features = len(feature_names)

# تولید داده‌ها (همان مقادیر قبلی)
X = np.column_stack([
    np.random.normal(120, 20, n_patients), np.random.normal(80, 10, n_patients),
    np.random.normal(200, 40, n_patients), np.random.normal(5.5, 1.0, n_patients),
    np.random.normal(25, 5, n_patients), np.random.normal(4.5, 0.5, n_patients),
    np.random.normal(7.0, 1.5, n_patients), np.random.normal(140, 10, n_patients),
    np.random.normal(4.0, 0.5, n_patients), np.random.normal(250, 50, n_patients),
    np.random.normal(50, 10, n_patients), np.random.normal(150, 30, n_patients),
    np.random.normal(3.0, 0.8, n_patients), np.random.normal(60, 15, n_patients),
    np.random.normal(98, 2, n_patients), np.random.normal(40, 5, n_patients),
    np.random.normal(9.5, 1.0, n_patients), np.random.normal(250, 50, n_patients),
    np.random.normal(10.0, 2.0, n_patients), np.random.normal(8.0, 1.5, n_patients),
    np.random.normal(30, 10, n_patients),
])
print(X.shape)

query = np.array([
    134.0, 85.0, 210.0, 5.7, 27.0, 4.8, 7.2, 138.0, 4.1, 260.0,
    48.0, 155.0, 3.2, 62.0, 97.0, 39.0, 10.0, 240.0, 11.0, 8.5, 35.0
])


# ==========================================
# 2. الگوریتم هوشمند با خروجی تحلیلی
# ==========================================
def smart_patient_finder(X, query, feature_names, k=10, n_iterations=5000, L=5):
    n_patients, n_features = X.shape

    # --- نرمال‌سازی (Z-Score) ---
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    X_scaled = (X - mean) / std
    query_scaled = (query - mean) / std

    # --- رأی‌گیری مونت‌کارلو ---
    votes = np.zeros(n_patients, dtype=int)
    for _ in range(n_iterations):
        subset_indices = np.random.choice(n_features, k, replace=False)
        # محاسبه فاصله اقلیدسی توان‌دار (سریع‌تر)
        dist_sq = np.sum((X_scaled[:, subset_indices] - query_scaled[subset_indices]) ** 2, axis=1)
        closest_idx = np.argmin(dist_sq)
        votes[closest_idx] += 1

    # پیدا کردن نفرات برتر
    top_indices = np.argsort(-votes)[:L]

    # --- ساخت گزارش تحلیلی (Pandas) ---
    results = []
    for idx in top_indices:
        # تحلیل: چرا این بیمار انتخاب شد؟
        # محاسبه اختلاف نرمال‌شده برای همه ویژگی‌ها
        diffs = np.abs(X_scaled[idx] - query_scaled)

        # پیدا کردن 3 ویژگی که کمترین اختلاف را دارند (بیشترین شباهت)
        best_match_indices = np.argsort(diffs)[:3]
        best_match_names = [feature_names[i] for i in best_match_indices]

        # ساخت ردیف دیتافریم
        results.append({
            "Patient ID": idx,
            "Similarity Score (Votes)": votes[idx],
            "Top 3 Matching Features": ", ".join(best_match_names),
            # محاسبه فاصله کلی برای اطمینان
            "Overall Deviation": np.round(np.sum(diffs), 2)
        })

    return pd.DataFrame(results)


# ==========================================
# 3. اجرا و نمایش
# ==========================================
start = time.time()
df_results = smart_patient_finder(X, query, feature_names, n_iterations=5000)
end = time.time()

print(f"Search completed in {end - start:.4f} seconds.\n")
print("=== Top Similar Patients Report ===")
print(df_results.to_string(index=False))

# نمایش جزئیات دقیق‌تر برای بهترین گزینه
best_patient_id = df_results.iloc[0]["Patient ID"]
print(f"\n--- Detailed Comparison for Patient {best_patient_id} ---")
print(f"{'Feature':<20} | {'Query Value':<12} | {'Patient Value':<12} | {'Diff'}")
print("-" * 55)

# مقایسه مقادیر واقعی (نه نرمال شده) برای درک پزشکی
for i, name in enumerate(feature_names):
    q_val = query[i]
    p_val = X[best_patient_id, i]
    # فقط ویژگی‌هایی که تفاوت کمی دارند را نشان بده (مثلاً زیر ۵ درصد خطا) یا ۵ تای اول لیست شباهت
    diff_percent = abs(q_val - p_val) / ((q_val + 0.001)) * 100
    if diff_percent < 5:  # فقط شباهت‌های خیلی زیاد را چاپ کن
        print(f"{name:<20} | {q_val:<12.2f} | {p_val:<12.2f} | {diff_percent:.1f}%")