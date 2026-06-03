# Project State

- Project type: new project.
- Completed work: initial repository scaffold created; project requirements reviewed; raw data files placed under `data/raw`; initial EDA completed for `customer_info` and `customer_basket`; relationship validation completed; data quality decisions documented; customer-level feature engineering completed; `customer_tenure` corrected to use the dataset reference year; preprocessing completed; `data/processed/customer_features_model.csv` created; advanced EDA and feature selection completed; `data/processed/selected_model_features.csv` created.
- Current phase: advanced EDA and feature selection complete.
- Recommended next task: clustering baseline.
- Known risks/issues: `selected_model_features.csv` keeps `customer_id` for traceability and contains a compact first-baseline feature set; location columns, redundant household fields, raw first transaction year, redundant basket totals, and duplicate binary gender encoding were removed; selected features are scaled/encoded from `customer_features_model.csv`; final clustering must include every `customer_id`; avoid overengineering.
