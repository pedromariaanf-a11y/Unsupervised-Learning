# Customer Segmentation

## Objective

Build a customer segmentation workflow that identifies meaningful customer groups, analyzes their behavior, proposes targeted promotions, and produces a final `customer_clusters.csv` with every `customer_id` and its assigned cluster.

## Datasets

- `customer_info.csv`: customer-level information.
- `customer_basket.csv`: basket and transaction-level information.

Raw datasets should be inspected before modelling and should not be modified directly.

## Setup

```bash
pip install -r requirements.txt
```

## Expected Flow

1. Initial EDA.
2. Feature engineering.
3. Preprocessing with the final compact RobustScaler feature set.
4. Advanced EDA with insights.
5. Feature selection / feature set design.
6. Final K-Means modelling.
7. Model comparison and robustness checks.
8. Basket profiling by final cluster.
9. Association rules by cluster.
10. Targeted promotions by cluster.
11. Final report and `customer_clusters.csv`.

## Repository Structure

```text
data/
  raw/
    customer_info.csv
    customer_basket.csv
  processed/
    customer_features_info.csv
    selected_model_features.csv
notebooks/
  01_initial_eda.ipynb
  02_feature_engineering.ipynb
  03_preprocessing.ipynb
  04_advanced_eda_feature_selection.ipynb
  05_kmeans_baseline.ipynb
  06_model_comparison.ipynb
  06b_kmeans_robustness_decision.ipynb
  06c_feature_set_decision.ipynb
  06d_final_preprocessing_decision.ipynb
  07_basket_profiling.ipynb
  08_association_rules_by_cluster.ipynb
  09_promotion_recommendations.ipynb
outputs/
  customer_clusters.csv
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
README.md
AGENTS.md
PROJECT_STATE.md
requirements.txt
```
