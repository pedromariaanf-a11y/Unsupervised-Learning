# Project State

- Project type: customer segmentation project.
- Current state: ready for K-Means clustering baseline.
- Completed work today: reviewed repository consistency; confirmed initial EDA, feature engineering, preprocessing, advanced EDA, and feature selection are complete; created `DECISIONS.md`; cleaned stale notebook wording/output in `notebooks/02_feature_engineering.ipynb`.
- Validated outputs: `data/processed/customer_features.csv`, `data/processed/customer_features_info.csv`, `data/processed/customer_features_model.csv`, and `data/processed/selected_model_features.csv` exist; customer-level outputs keep 33,038 customers; processed outputs have no duplicated `customer_id`; `customer_tenure` has no negative values; `selected_model_features.csv` has no missing values and excludes customer_basket-derived features.
- Known decisions/risks: `customer_info` is the base customer table; every customer must be preserved; `loyalty_card_number` is represented as `has_loyalty_card`; `customer_name` is excluded except for `degree_level`; the dataset maximum `year_first_transaction` is used as the reference year for `age` and `customer_tenure`; basket-derived features are useful for profiling and promotions but excluded from the first clustering baseline because 4,911 customers have no basket records; location features were excluded from the first selected baseline to avoid dominating distance-based clustering.
- Recommended next task: create the K-Means clustering baseline notebook using `data/processed/selected_model_features.csv`.
- Blockers: none found.
