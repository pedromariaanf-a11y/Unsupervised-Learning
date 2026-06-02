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
3. Preprocessing.
4. Advanced EDA with insights.
5. Feature selection / feature set design.
6. Modelling.
7. Cluster profiling.
8. Targeted promotions.
9. Final report and `customer_clusters.csv`.

## Repository Structure

```text
data/
  raw/
  processed/
notebooks/
outputs/
  figures/
src/
README.md
AGENTS.md
PROJECT_STATE.md
requirements.txt
```
