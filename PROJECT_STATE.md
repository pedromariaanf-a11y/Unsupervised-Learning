# Project State

- Project type: new project.
- Completed work: initial repository scaffold created; project requirements reviewed; raw data files placed under `data/raw`; initial EDA completed for `customer_info` and `customer_basket`; relationship validation completed; data quality decisions documented; customer-level feature engineering completed; `customer_tenure` corrected to use the dataset reference year; preprocessing completed; `data/processed/customer_features_model.csv` created.
- Current phase: preprocessing complete.
- Recommended next task: advanced EDA / feature selection before modelling.
- Known risks/issues: `customer_features_model.csv` keeps `customer_id` for traceability but excludes it from scaled model features; `most_frequent_product` is excluded from model features; missing numeric values are filled with medians, missing categorical values with `Unknown`, promotion percentages are clipped to `0` to `100`, `customer_gender` and `degree_level` are one-hot encoded, and numeric model features are scaled with `StandardScaler`; final clustering must include every `customer_id`; avoid overengineering.
