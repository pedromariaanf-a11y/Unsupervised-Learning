# Customer Segmentation — Unsupervised Learning

End-to-end retail customer segmentation using **K-Means clustering** on 33,038 customers, enriched with basket analysis, association rules, and targeted promotion recommendations.

---

## Overview

| Item | Detail |
|---|---|
| **Dataset** | 33,038 customers · 28,127 with basket records |
| **Model** | `KMeans(n_clusters=5, random_state=42, n_init=50)` |
| **Scaler** | `RobustScaler` |
| **Feature set** | 20 compact business features (demographic + behavioural spend shares) |
| **Metrics** | Silhouette 0.2208 · Calinski-Harabasz 6,347 · Davies-Bouldin 1.54 |

---

## Customer Segments

| Cluster | Persona | Customers | Share |
|:---:|---|---:|---:|
| 0 | Promo-Sensitive Older Tech Snackers | 2,619 | 7.9% |
| 1 | Mainstream Grocery Families | 18,385 | 55.6% |
| 2 | Vegetarian-Leaning Family Households | 5,538 | 16.8% |
| 3 | Demanding Personal-Care and Tech Shoppers | 3,552 | 10.8% |
| 4 | High-Value Large Loyal Families | 2,944 | 8.9% |

---

## Pipeline

```
Raw data → EDA → Feature Engineering → Preprocessing
       → K-Means baseline → Model comparison (KMeans / GMM / Agglomerative / DBSCAN)
       → Robustness & feature-set sensitivity checks → Final model
       → Basket profiling → Association rules → Promotion recommendations
```

**Notebooks run in order:**

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_initial_eda.ipynb` | Exploratory analysis of raw data |
| 02 | `02_feature_engineering.ipynb` | Build customer-level feature table |
| 03 | `03_preprocessing.ipynb` | Scale and select final model features |
| 04 | `04_advanced_eda_feature_selection.ipynb` | Feature selection analysis |
| 05 | `05_kmeans_baseline.ipynb` | K-Means sweep k=2..8, elbow + metrics |
| 06 | `06_model_comparison.ipynb` | KMeans vs GMM vs Agglomerative vs DBSCAN |
| 06b | `06b_kmeans_robustness_decision.ipynb` | Seed-stability ARI check |
| 06c | `06c_feature_set_decision.ipynb` | Compact vs full feature-set sensitivity |
| 06d | `06d_final_preprocessing_decision.ipynb` | Scaler sensitivity (Robust / Standard / Log) |
| 07 | `07_basket_profiling.ipynb` | Post-cluster basket profiling |
| 08 | `08_association_rules_by_cluster.ipynb` | Pairwise association rules per cluster |
| 09 | `09_promotion_recommendations.ipynb` | Targeted promotions per persona |

---

## Key Design Decisions

- **Basket data excluded from clustering** — 4,911 customers have no basket records; zero values would mean "not observed", not true zero behaviour. Basket data is used only *after* cluster assignment for profiling and promotions.
- **Compact feature set** — gender and degree dummy columns were removed after feature-set sensitivity analysis showed the 20 numeric business features produce better-separated clusters.
- **k=5 vs k=6** — k=6 splits the large mainstream cluster into two readable subgroups but has weaker internal metrics. k=5 is kept as the metric-supported final model; the k=6 split is documented as a business tradeoff.

See [`DECISIONS.md`](DECISIONS.md) for the full decision trail.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

---

## Repository Structure

```text
data/
  raw/
    customer_info.csv          ← raw customer demographics + behaviour
    customer_basket.csv        ← transaction-level basket records
  processed/
    customer_features_info.csv ← engineered customer feature table
    selected_model_features.csv← final scaled model input (33,038 × 21)
notebooks/                     ← numbered pipeline notebooks (run in order)
outputs/
  customer_clusters.csv        ← final deliverable: customer_id + cluster
  kmeans_metrics.csv
  kmeans_cluster_profile_summary.csv
  kmeans_cluster_spending_shares.csv
  model_comparison_metrics.csv
  model_comparison_cluster_balance.csv
  segmentation_robustness_metrics.csv
  segmentation_seed_stability.csv
  segmentation_robustness_recommendation.csv
  k5_k6_big_cluster_split_profile.csv
  feature_set_sensitivity_metrics.csv
  feature_set_sensitivity_recommendation.csv
  cluster_basket_profile.csv
  cluster_top_products.csv
  cluster_association_rules.csv
  cluster_promotion_recommendations.csv
src/
  clustering.py    ← KMeans/GMM fitting, metrics, seed stability, t-SNE
  features.py      ← customer info + basket feature engineering
  preprocessing.py ← cleaning, feature selection, RobustScaler pipeline
  promotions.py    ← basket profiling, association rules, promotion tables
report.md          ← full narrative project report
DECISIONS.md       ← documented analytical decision trail
requirements.txt
```
