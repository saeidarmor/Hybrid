import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans  # اضافه شد


def kmeans_scoring_algorithm(X, query, k_scoring=None, L=5, n_clusters=5, n_samples=2000):
    n, m = X.shape

    # اطمینان از دوبعدی بودن query
    if query.ndim == 1:
        query_rescaled = query.reshape(1, -1)
    else:
        query_rescaled = query

    if k_scoring is None:
        k_scoring = m // 2

    # ۱. خوشه‌بندی
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # ۲. پیدا کردن خوشه بیمار جدید
    query_cluster = kmeans.predict(query_rescaled)[0]

    # ۳. فیلتر کردن داده‌های همان خوشه
    cluster_mask = (cluster_labels == query_cluster)
    X_cluster = X[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]

    print(f"[K-Means] بیمار جدید به خوشه {query_cluster} تعلق دارد.")
    print(f"[K-Means] تعداد بیماران در این خوشه: {len(X_cluster)}")

    if len(X_cluster) == 0:
        raise ValueError("هیچ بیماری در خوشه مورد نظر یافت نشد!")

    # ۴. فراخوانی تابع امتیازدهی (این تابع باید تعریف شده باشد)
    # توجه: خروجی‌های این تابع را بر اساس منطق خودتان چک کنید
    try:
        top_local, votes_local = improved_scoring_algorithm(
            X_cluster, query, k=k_scoring, L=L, n_samples=n_samples, normalize=True
        )
    except NameError:
        print("خطا: تابع improved_scoring_algorithm تعریف نشده است!")
        return None, None

    # ۵. نگاشت ایندکس‌های محلی به جهانی
    top_global = cluster_indices[top_local]
    votes_global = np.zeros(n, dtype=int)
    votes_global[cluster_indices] = votes_local

    return top_global, votes_global


# --- تست کد ---
# ایجاد داده‌های فرضی برای اجرا شدن کد
X = np.random.rand(100, 10)
query = np.random.rand(10)


# تعریف تابع فرضی برای جلوگیری از خطا در اجرا
def improved_scoring_algorithm(X_c, q, k, L, n_samples, normalize):
    # یک پیاده‌سازی ساده برای تست
    scores = np.random.randint(0, 100, len(X_c))
    top_idx = np.argsort(scores)[-3:]  # ۳ تا از بهترین‌ها
    return top_idx, scores


print("=== 3. الگوریتم جدید: K-Means + Scoring ===")
t0 = time.time()
top3, votes3 = kmeans_scoring_algorithm(X, query, k_scoring=10, L=5, n_clusters=5, n_samples=2000)

if top3 is not None:
    print("بیماران مشابه (ایندکس):", top3)
    print("زمان اجرا:", f"{time.time() - t0:.4f} ثانیه")