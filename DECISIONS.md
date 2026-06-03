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
- Reason: It is scaled, encoded, compact, and excludes risky or redundant first-baseline features.
- Impact: The next phase can focus on clustering baseline design and evaluation without revisiting feature selection first.
