# Customer Segmentation Using Unsupervised Learning

*Machine Learning II — Group Project, Nova IMS*

---

## Overview

This project applies unsupervised machine learning to identify groups of customers with similar demographic and purchasing characteristics in a retail dataset. The goal was to produce a stable, interpretable segmentation that could inform targeted promotional strategies.

The work was completed as a university group project for the Machine Learning II course. The repository contains all the analysis steps, from initial data exploration through to cluster profiling, association rule analysis, and promotion recommendations.

---

## Objectives

- Explore and understand two retail datasets covering customer demographics and transaction-level basket records
- Engineer a meaningful set of features for clustering from raw customer data
- Evaluate multiple clustering algorithms and select a final model based on internal metrics and practical interpretability
- Profile each identified segment using demographic features, spending patterns, and basket behaviour
- Propose targeted promotional strategies for each segment, grounded in the data

---

## Dataset

Two raw datasets were used. Neither was modified directly; all transformations were applied in dedicated processing notebooks.

**`customer_info.csv`** contains one row per customer (33,038 unique customers). Fields include demographic information (age, gender, education prefix), household composition (number of children at home), loyalty card status, spending by product category, promotion usage rate, number of complaints, and year of first transaction.

**`customer_basket.csv`** contains transaction-level basket records. Each row is a single shopping basket linked to a customer. The basket dataset covers 28,127 of the 33,038 customers; the remaining 4,911 customers have no observed basket records. The catalogue spans 164 unique products, and basket sizes range from 1 to 18 items (median approximately 9).

An important constraint: because basket data is missing for roughly 15% of customers, basket-derived features were excluded from the clustering input. Including them would require imputing records for customers who may simply not have shopped during the observation period, which risks introducing fabricated signal into the model. Basket data was instead used after clustering for segment profiling and promotion design.

---

## Methodology

### Exploratory Data Analysis

Initial EDA examined distributions, missing values, spending patterns, and basket behaviour. Key findings included:

- Lifetime spending columns are right-skewed, as is typical for retail data
- `loyalty_card_number` behaves as a binary indicator — values are either 1.0 or missing — not a continuous number
- `percentage_of_products_bought_promotion` contained a small number of values outside the valid [0, 1] range, which were capped during preprocessing
- The most frequently purchased products across all baskets were asparagus and airpods, reflecting the dataset's unusually diverse product catalogue

### Feature Engineering

The following features were constructed from the raw data:

| Feature | Description |
|---|---|
| `age` | Derived from birth year and the dataset's maximum reference year |
| `customer_tenure` | Years since first transaction, anchored to the dataset's own maximum year |
| `has_loyalty_card` | Binary indicator derived from `loyalty_card_number` |
| `total_children_home` | Sum of `kids_home` and `teens_home` |
| `has_children` | Binary indicator for any children at home |
| `total_lifetime_spend` | Sum of all category spend columns |
| `share_*` | Proportion of total lifetime spend in each product category |

Customer names were excluded from modelling. A `degree_level` field was extracted from name prefixes (Bsc., Msc., Phd.) and retained for EDA but dropped from the final clustering input after feature sensitivity analysis showed it did not improve separation.

### Preprocessing

The final model input contains `customer_id` plus 20 numeric features scaled with `RobustScaler`. This scaler was chosen because spending columns are right-skewed and contain outliers; it uses the interquartile range and is therefore less sensitive to extreme values than `StandardScaler`. A dedicated sensitivity check across StandardScaler, MinMaxScaler, and RobustScaler on k=5 K-Means confirmed that RobustScaler produced the best silhouette and Davies-Bouldin scores on this dataset.

Gender and degree dummy columns were excluded from the final input after feature set sensitivity analysis showed the compact numeric business features produced better-separated clusters.

### Clustering

Four algorithm families were evaluated on the same 20-feature scaled input.

**K-Means** was tested for k = 2 through 10 using the elbow method, silhouette scores, Calinski-Harabász, and Davies-Bouldin. A reproducible 10,000-customer sample was used for silhouette calculation. `n_init=50` was used throughout to reduce initialisation sensitivity.

**Gaussian Mixture Models (GMM)** with diagonal covariance were tested at k = 4, 5, and 6.

**Agglomerative Clustering** (Ward linkage, k = 4, 5, 6) was evaluated on a reproducible 5,000-customer sample.

**DBSCAN** was tested across a grid of eps and min_samples values on the same 5,000-customer sample.

### Model Selection

K-Means k=5 was selected as the final model. The reasoning is documented in detail in [`DECISIONS.md`](DECISIONS.md).

In brief: GMM models produced lower silhouette scores and worse cluster balance than K-Means k=5; Agglomerative Clustering did not outperform K-Means on any metric; DBSCAN produced a single dominant cluster (>95% of the sample) under all tested configurations, indicating that the customer population does not have strong density-based structure. DBSCAN is noted in the project as useful for outlier diagnostics, not as a primary segmentation approach.

---

## Results

### Model Performance

| k | Silhouette | Calinski-Harabász | Davies-Bouldin |
|:---:|---:|---:|---:|
| 2 | 0.364 | 8,195 | 1.309 |
| 3 | 0.274 | 7,124 | 1.299 |
| 4 | 0.239 | 7,018 | 1.584 |
| **5 (selected)** | **0.221** | **6,347** | **1.539** |
| 6 | 0.152 | 5,767 | 1.854 |

The silhouette score at k=2 is higher, but a two-cluster solution does not produce commercially useful segments. k=5 provides the best balance between metric quality and the number of interpretable, actionable groups.

The silhouette score of 0.221 is modest, which is common for real retail customer data where purchasing behaviour is continuous and overlapping rather than cleanly separable. The project report discusses this honestly in the limitations section.

**Seed stability.** Adjusted Rand Index values across five different random seeds ranged from 0.999 to 1.000, confirming that the k=5 solution is stable and not sensitive to initialisation.

**Algorithm comparison.**

| Model | Silhouette | Min cluster % | Max cluster % |
|---|---:|---:|---:|
| K-Means k=5 | **0.221** | 7.9% | 55.6% |
| GMM diagonal k=4 | 0.101 | 10.3% | 53.9% |
| GMM diagonal k=5 | 0.121 | 10.5% | 41.7% |
| GMM diagonal k=6 | 0.108 | 9.6% | 26.7% |

### Customer Segments

| Cluster | Persona | Customers | Share |
|:---:|---|---:|---:|
| 0 | Promo-Sensitive Older Tech Snackers | 2,619 | 7.9% |
| 1 | Mainstream Grocery Families | 18,385 | 55.6% |
| 2 | Vegetarian-Leaning Family Households | 5,538 | 16.8% |
| 3 | Demanding Personal-Care and Tech Shoppers | 3,552 | 10.8% |
| 4 | High-Value Large Loyal Families | 2,944 | 8.9% |

**Cluster 0** — Older customers (mean age 60.4) with the highest promotion usage rate (45.1%) and the lowest lifetime spend (mean £10,335). Electronics and snack products appear prominently alongside grocery purchases. Loyalty card penetration is 67.7%.

**Cluster 1** — The largest and most heterogeneous group, covering 55.6% of the dataset. Spending is heavily concentrated in groceries (75.9% of lifetime spend). This segment represents a broad mainstream base; the project acknowledges that its size reflects genuine customer heterogeneity rather than a modelling weakness.

**Cluster 2** — Characterised by the highest vegetable spending share (12.7%) and the lowest meat spending share (0.8%) of all clusters. Basket coverage is the highest of all segments (89.1%), suggesting consistent shopping engagement. Promotion usage is low (13.3%).

**Cluster 3** — The highest electronics spending share by a significant margin (37.5% of lifetime spend). This cluster also has the lowest loyalty card penetration (49.2%) and the highest complaint rate (1.08 per customer), which is discussed in the project as a potential churn signal. Fewer children at home (mean 0.7) than other segments.

**Cluster 4** — The highest-spending segment (mean lifetime spend £37,486) with the largest household size (mean 6.4 children at home) and the longest customer tenure. Promotion usage is relatively low (21.5%), consistent with need-driven rather than discount-driven purchasing.

---

## Business Insights

The following recommendations are proposals derived from the cluster profiles and association rule analysis. They have not been validated in a real retail environment.

| Cluster | Proposed Promotion | Rationale |
|:---:|---|---|
| 0 | Loyalty-card discount bundle on tech accessories + snacks | High promotion sensitivity; observed co-purchase between headphones and energy drinks (lift 2.57) |
| 1 | Multi-buy offer on fresh produce | Grocery-dominant spend; strong salad–avocado co-purchase pattern (lift 2.83) |
| 2 | Curated healthy family basket | Low meat, high vegetable spend; consistent shoppers who respond to product curation more than price cuts |
| 3 | Retention bundle: personal-care replenishment | Low loyalty, high complaints; recurring household purchases provide a reason to stay, with a technology cross-sell opportunity (headphones–airpods lift 3.83) |
| 4 | Premium loyalty reward tied to breakfast staples | Highest-value segment; deep discounting is inappropriate; reward-based retention is more suitable |

The association rules used to inform these recommendations describe product co-occurrence, not causality. A product pair with a high lift value appears together more often than chance would predict — it does not mean purchasing one item causes the purchase of the other. The rules are treated as affinity signals rather than causal mechanisms.

---

## Technologies

- **Python 3.13**
- **pandas** — data manipulation and feature engineering
- **numpy** — numerical operations
- **scikit-learn** — K-Means, GMM, Agglomerative Clustering, DBSCAN, RobustScaler, evaluation metrics
- **matplotlib** and **seaborn** — visualisation

Full dependency versions are listed in [`requirements.txt`](requirements.txt).

---

## Repository Structure

```
data/
  raw/                         raw CSV files (not modified)
  processed/                   engineered feature tables
notebooks/                     analysis notebooks, numbered in pipeline order
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
  customer_clusters.csv        final deliverable: customer_id and cluster label for all 33,038 customers
  cluster_promotion_recommendations.csv
  cluster_association_rules.csv
  (other metric and profile tables)
src/
  clustering.py                K-Means/GMM fitting, metrics, seed stability, t-SNE projection
  features.py                  customer info and basket feature engineering
  preprocessing.py             cleaning, feature selection, RobustScaler pipeline
  promotions.py                basket profiling, association rules, promotion table assembly
report.md                      full project report
DECISIONS.md                   documented analytical decisions and reasoning
requirements.txt
```

---

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

The notebooks are numbered and designed to be run in order, starting with `01_initial_eda.ipynb`. Each notebook reads its inputs from `data/` or `outputs/` using direct `pd.read_csv` calls and saves its outputs as flat CSV files to `outputs/`.

The raw datasets are included in `data/raw/`. The processed feature tables (`data/processed/`) and all output files are also committed, so it is possible to open any individual notebook and inspect the results without running the full pipeline from scratch.

---

## Limitations

- **Cluster 1 breadth.** The largest segment contains 55.6% of all customers. This reflects genuine heterogeneity in the data, not a modelling failure. A k=6 alternative that splits this segment is documented in the project but was not selected as the final model because it produces meaningfully weaker internal metrics (silhouette 0.152 vs 0.221).

- **Modest silhouette score.** A silhouette of 0.221 is typical for real retail data where customer behaviour does not form sharp, well-separated groups. The metric is discussed honestly in the report alongside the reasons for choosing this model.

- **Basket data coverage.** 4,911 of the 33,038 clustered customers have no basket records. These customers retain their cluster assignment in `outputs/customer_clusters.csv` but cannot be characterised by basket behaviour.

- **Association rules describe co-occurrence.** Lift values indicate that product pairs appear together more often than chance — they do not establish causality. Promotion recommendations should be treated as hypotheses to validate, not as proven strategies.

- **No external validation.** This is an academic project. The cluster assignments and promotion recommendations have not been tested against actual customer behaviour or business outcomes.

---

## Academic Context

This project was completed as a group assignment for the Machine Learning II course at Nova IMS. The goal was to apply unsupervised learning methods to a realistic dataset and produce a result that could be meaningfully interpreted in a business context.

The full analysis, including the reasoning behind all major methodological choices, is documented in [`DECISIONS.md`](DECISIONS.md) and [`report.md`](report.md).
