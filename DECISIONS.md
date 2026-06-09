# Decisions

## 1. Base Clustering Dataset

- Decision: Use `customer_info` as the base clustering dataset.
- Reason: `customer_info` is the customer-level table and contains every customer required in the final output.
- Impact: All customer-level outputs should start from `customer_info` and preserve one row per `customer_id`.

## 2. Preserve All Customers

- Decision: Preserve all customers in every customer-level output.
- Reason: The final clustering deliverable must include every `customer_id`.
- Impact: Merges from other sources should be left joins onto the customer-level base.

## 3. Loyalty Card Feature

- Decision: Treat `loyalty_card_number` as a binary `has_loyalty_card` feature.
- Reason: The raw value behaves like an indicator, and missing values likely mean no loyalty card.
- Impact: Do not use `loyalty_card_number` as a raw numeric modelling feature.

## 4. Customer Name

- Decision: Extract `degree_level` from visible `customer_name` prefixes, but do not use `customer_name` directly.
- Reason: Names are identifiers, while prefixes such as `Bsc.`, `Msc.`, and `Phd.` may contain a simple customer context signal.
- Impact: Keep `degree_level`; exclude raw names from modelling features.

## 5. Dataset Reference Year

- Decision: Use the dataset maximum `year_first_transaction` as the reference year for `customer_tenure` and `age`.
- Reason: The features should be based on the dataset timeline, not the current calendar year.
- Impact: `customer_tenure` has no negative values in the current processed customer-info feature table.

## 6. Basket Features

- Decision: Keep basket-derived features for profiling and promotions, but exclude them from the first clustering baseline.
- Reason: `customer_basket` does not cover all customers; 4,911 customers have no basket records, and zero basket values may mean "not observed" rather than true zero behaviour.
- Impact: Basket features remain useful for EDA and cluster interpretation, but are not included in `selected_model_features.csv`.

## 7. First K-Means Input

- Decision: Use `data/processed/selected_model_features.csv` as the input for the first K-Means baseline.
- Reason: It is scaled, compact, and excludes risky or redundant first-baseline features.
- Impact: The next phase can focus on clustering baseline design and evaluation without revisiting feature selection first.

## 8. Model Candidate After Comparison

- Decision: Keep K-Means k=5 as the strongest practical candidate for the final customer segmentation model.
- Reason: K-Means k=5 performed better than the tested GMM alternatives on the main internal metrics; it produced more usable cluster balance than the GMM models; Agglomerative Clustering did not outperform it; DBSCAN did not provide a stable multi-segment solution and is better treated as an outlier diagnostic.
- Alternatives considered: GMM diagonal models with 4, 5, and 6 components; Agglomerative Clustering with 4, 5, and 6 clusters on a sample; DBSCAN with eps/min_samples grid on a sample.
- Impact: The final `outputs/customer_clusters.csv` uses K-Means k=5; basket data should be used after clustering for profiling, association rules, and promotion design; DBSCAN should not be used as the main segmentation output.

## 9. Final Segmentation Model

- Decision: Use the compact business feature set with `RobustScaler` and `KMeans(n_clusters=5, random_state=42, n_init=50)` as the final segmentation model.
- Reason: The compact set removes gender and degree dummy columns, keeps the business-readable numeric customer features, and performed better than the previous scaled feature set during feature/scaler sensitivity checks.
- Impact: `data/processed/selected_model_features.csv` should contain `customer_id` plus 20 compact model features; gender and degree dummy columns should not be included in the final clustering input; `outputs/customer_clusters.csv` should be regenerated from this final input.
- Basket policy: Basket data is excluded from clustering and used only after cluster assignment for basket profiling, association rules, and promotion design.
- Evidence notebooks: `06b_kmeans_robustness_decision.ipynb`, `06c_feature_set_decision.ipynb`, and `06d_final_preprocessing_decision.ipynb`.
