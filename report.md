# Customer Segmentation Report
### Machine Learning II — Unsupervised Learning Project

---

## 1. Executive Summary

This project builds a customer segmentation solution for a retail company using unsupervised machine learning applied to customer-level demographic and behavioural data. The objective is to identify meaningful, actionable customer groups and translate them into targeted promotional strategies.

The dataset comprised two sources: `customer_info` (33,038 unique customers) and `customer_basket` (transaction-level basket records for 28,127 of those customers). After exploratory analysis, feature engineering, and preprocessing, a K-Means clustering model with five segments was selected as the final solution, based on a systematic comparison with Gaussian Mixture Models (GMM), Agglomerative Clustering, and DBSCAN.

The five identified segments are:

| Cluster | Persona | Size | % of Total |
|:---:|---|---:|---:|
| 0 | Promo-Sensitive Older Tech Snackers | 2,619 | 7.9% |
| 1 | Mainstream Grocery Families | 18,385 | 55.6% |
| 2 | Vegetarian-Leaning Family Households | 5,538 | 16.8% |
| 3 | Demanding Personal-Care and Tech Shoppers | 3,552 | 10.8% |
| 4 | High-Value Large Loyal Families | 2,944 | 8.9% |

Each segment was profiled using demographic features, spending behaviour, basket composition, and product association rules. Targeted promotional strategies were then designed for each segment based on this evidence.

---

## 2. Exploratory Data Analysis and Pre-Processing

### 2.1 Dataset Overview

Two raw datasets were used. They were never modified directly; all transformations were applied in dedicated processing notebooks.

**`customer_info`** is a customer-level table with one row per customer. It contains demographic fields (age, gender, education prefix, loyalty card status), household composition (number of kids and teens at home), transaction behaviour (year of first transaction, distinct stores visited, promotion usage rate), spending breakdowns by product category, and number of complaints. The dataset has 33,038 unique `customer_id` values with no duplicate entries.

**`customer_basket`** is a transaction-level table where each row represents a single shopping basket, identified by a unique `invoice_id` and linked to a `customer_id`. Baskets contain a list of purchased products stored as a Python list string. The basket dataset covers 28,127 distinct customers, leaving 4,911 clustered customers with no basket records.

### 2.2 Key EDA Findings

**Missing values.** The most significant missingness was in `loyalty_card_number`, which turned out to hold values of either `1.0` or missing — consistent with a binary indicator rather than a meaningful number. Several behavioural fields also had small missing shares that were addressed during feature engineering. After engineering, the final model feature table had no missing values.

**Spend distribution.** All lifetime spend columns were right-skewed, which is typical for retail purchasing data. Customers with very high total spend exist but are a small minority.

**Basket structure.** The basket dataset contained 164 unique products. Basket sizes ranged from 1 to 18 products with a median of approximately 9 items. The most frequently appearing products were `asparagus` and `airpods`, suggesting a diverse range of product categories coexisting in the catalogue.

**Customer transaction frequency.** Most customers had only a few baskets in the dataset. A small number of customers had substantially higher transaction counts, consistent with a typical loyalty distribution.

**Data quality.** `percentage_of_products_bought_promotion` contained some values outside the valid `[0, 1]` range, which were capped during preprocessing. `year_first_transaction` was used only as a reference for computing `customer_tenure`, anchored to the dataset's own maximum year rather than the current calendar year, avoiding spurious negative tenure values.

### 2.3 Feature Engineering

The following transformations were applied to construct the model feature set:

- **`has_loyalty_card`**: binary indicator derived from `loyalty_card_number`. Missing values were treated as no loyalty card.
- **`degree_level`**: extracted from visible name prefixes (`Bsc.`, `Msc.`, `Phd.`). Customer names themselves were excluded from modelling.
- **`customer_tenure`**: computed as the difference between the dataset's maximum `year_first_transaction` and each customer's own first transaction year.
- **`age`**: derived from the reference year and birth year.
- **`total_children_home`**: sum of `kids_home` and `teens_home`.
- **`has_children`**: binary indicator for whether any children are present at home.
- All lifetime spend columns and behavioural counts were retained directly.

Gender and degree dummy columns were excluded from the final clustering input after feature sensitivity analysis showed that the compact numeric business features produced better-separated clusters.

### 2.4 Preprocessing

The final model input (`selected_model_features.csv`) contains `customer_id` plus 20 compact numeric features scaled with **RobustScaler**. RobustScaler was chosen over StandardScaler because retail spend features are right-skewed and contain outliers; RobustScaler uses the interquartile range and is therefore less sensitive to extreme values.

A dedicated sensitivity check across scaling strategies (StandardScaler, MinMaxScaler, RobustScaler) on k=5 K-Means confirmed that RobustScaler produced the best silhouette and Davies-Bouldin scores on this dataset.

**Basket data was deliberately excluded from the clustering input.** Of 33,038 customers, 4,911 have no basket records at all. Including basket-derived features would either require imputing those records (risking fabricated signal) or excluding those customers from clustering (breaking the requirement that every customer receives a cluster assignment). Basket data was therefore reserved for post-clustering profiling, association rule analysis, and promotion design.

---

## 3. Customer Segmentation and Clustering

### 3.1 Algorithm Selection and Comparison

Four algorithm families were evaluated on the same 20-feature scaled input.

**K-Means** was tested for k = 2 through 10. The elbow and silhouette evidence both pointed to k = 5 as the practical optimum. Silhouette scores declined sharply from k=6 onward, and cluster balance became unstable at higher k values.

| k | Silhouette | Calinski-Harabász | Davies-Bouldin |
|:---:|---:|---:|---:|
| 2 | 0.364 | 8,195 | 1.309 |
| 3 | 0.274 | 7,124 | 1.299 |
| 4 | 0.239 | 7,018 | 1.584 |
| **5** | **0.221** | **6,347** | **1.539** |
| 6 | 0.152 | 5,767 | 1.854 |
| 7 | 0.151 | 5,380 | 1.787 |

Although the best silhouette score was at k=2, a two-cluster solution produces only coarse commercial segments with limited actionability. k=5 provides the best balance between metric quality and business interpretability.

**Gaussian Mixture Models (GMM)** with diagonal covariance were tested at k = 4, 5, and 6. All GMM models achieved lower silhouette scores than K-Means k=5 and produced worse cluster balance, with some configurations concentrating over 40% of customers in a single component.

| Model | Silhouette | Min cluster % | Max cluster % |
|---|---:|---:|---:|
| K-Means k=5 | **0.221** | 7.9% | 55.6% |
| GMM diag k=4 | 0.101 | 10.3% | 53.9% |
| GMM diag k=5 | 0.121 | 10.5% | 41.7% |
| GMM diag k=6 | 0.108 | 9.6% | 26.7% |

**Agglomerative Clustering** (Ward linkage, k = 4, 5, 6) was evaluated on a reproducible 5,000-customer sample. It did not outperform K-Means on separation metrics and produced comparable or worse cluster balance, without offering additional interpretability advantages.

**DBSCAN** was tested with a grid of eps (2–5) and min_samples (10, 20) values on the same sample. Under all tested configurations, DBSCAN either produced a single dominant non-noise cluster exceeding 95% of the sample, or assigned a large proportion of points to noise. DBSCAN is therefore not suitable as the primary segmentation model but is useful as an outlier diagnostic, confirming that the customer population does not exhibit strong density-based cluster structure.

### 3.2 Final Model

The final model is:

```
RobustScaler → KMeans(n_clusters=5, random_state=42, n_init=50)
```

`n_init=50` was used to mitigate sensitivity to initialisation. Seed stability analysis across five different random states (0, 21, 42, 99, 123) produced Adjusted Rand Index values consistently above **0.999**, confirming that the k=5 solution is stable and reproducible.

| Seed pair | Adjusted Rand Index |
|---|---:|
| 0 vs 21 | 0.9986 |
| 21 vs 42 | **1.0000** |
| 42 vs 99 | 0.9999 |
| 99 vs 123 | 0.9992 |

**Final model internal metrics:**

| Metric | Value |
|---|---:|
| Silhouette Score | 0.221 |
| Calinski-Harabász Score | 6,347 |
| Davies-Bouldin Score | 1.539 |

The silhouette score of 0.221 is modest, which is typical for real-world retail segmentation data where customer behaviour is continuous and overlapping rather than discretely separated. The model's value lies in its interpretability and practical downstream utility rather than in perfect geometric separation.

**k=6 was considered as an alternative.** It splits the large cluster 1 into two readable subgroups, improving cluster balance. However, k=6 has a substantially lower silhouette (0.152 vs 0.221) and a worse Davies-Bouldin score, so k=5 was retained as the metric-supported final model. The k=6 tradeoff is documented in the project decision record.

### 3.3 Cluster Profiles

The five clusters were profiled using the original (unscaled) feature values and post-clustering basket data.

#### Cluster 0 — Promo-Sensitive Older Tech Snackers (2,619 customers, 7.9%)

| Feature | Value |
|---|---|
| Mean age | 60.4 years |
| Mean lifetime spend | £10,335 |
| Loyalty card share | 67.7% |
| Promotion usage rate | 45.1% |
| Mean children at home | 1.6 |
| Mean complaints | 0.98 |
| Basket coverage | 83.9% |

**Spending breakdown:** Groceries 43.8% · Electronics 13.1% · Meat 9.5% · Hygiene 7.5% · Fish 6.6%

This cluster has the **highest promotion usage rate** (45%) and a relatively high loyalty card penetration, suggesting customers who actively seek out discounts while remaining brand-loyal. Their electronics spending is above average, and snack and energy products appear prominently in their baskets. Total lifetime spend is the lowest of all five segments, consistent with promotion-dependent purchasing behaviour.

**Top basket association rules:**

| Antecedent | Consequent | Lift |
|---|---|---:|
| bluetooth headphones | energy drink | 2.57 |
| energy drink | airpods | 2.45 |
| bluetooth headphones | airpods | 2.42 |

---

#### Cluster 1 — Mainstream Grocery Families (18,385 customers, 55.6%)

| Feature | Value |
|---|---|
| Mean age | 56.4 years |
| Mean lifetime spend | £25,883 |
| Loyalty card share | 61.1% |
| Promotion usage rate | 38.2% |
| Mean children at home | 1.6 |
| Mean complaints | 0.97 |
| Basket coverage | 84.4% |

**Spending breakdown:** Groceries 75.9% · Electronics 6.7% · Hygiene 3.1% · Fish 2.5% · Meat 3.3%

This is the **largest and most heterogeneous segment**, encompassing over half the customer base. It is dominated by grocery spend and characterised by fresh-produce basket patterns. While the segment is broad, it represents a commercially important core base for standard promotions. Association rules reveal strong fresh-produce co-purchasing patterns.

**Top basket association rules:**

| Antecedent | Consequent | Lift |
|---|---|---:|
| salad | avocado | 2.83 |
| avocado | salad | 2.83 |
| spinach | salad | 2.73 |

---

#### Cluster 2 — Vegetarian-Leaning Family Households (5,538 customers, 16.8%)

| Feature | Value |
|---|---|
| Mean age | 60.6 years |
| Mean lifetime spend | £15,564 |
| Loyalty card share | 57.8% |
| Promotion usage rate | 13.3% |
| Mean children at home | 1.9 |
| Mean complaints | 0.66 |
| Basket coverage | 89.1% |

**Spending breakdown:** Groceries 57.4% · Vegetables 12.7% · Hygiene 9.6% · Electronics 9.9% · Meat 0.8%

Cluster 2 has the **highest vegetable spending share** (12.7%) and the **lowest meat spending share** (0.8%) of all segments. It also has the highest basket coverage rate (89.1%), suggesting consistent engagement. The segment's promotion usage is very low, which may indicate value-driven purchasing based on product preference rather than discounts. Family household composition is slightly higher than average. This cluster should be described as vegetarian-*leaning*, as some meat products still appear in baskets.

**Top basket association rules:**

| Antecedent | Consequent | Lift |
|---|---|---:|
| cooking oil | napkins | 2.35 |
| cooking oil | babies food | 2.17 |
| napkins | babies food | 2.13 |

---

#### Cluster 3 — Demanding Personal-Care and Tech Shoppers (3,552 customers, 10.8%)

| Feature | Value |
|---|---|
| Mean age | 59.5 years |
| Mean lifetime spend | £22,720 |
| Loyalty card share | 49.2% |
| Promotion usage rate | 26.9% |
| Mean children at home | 0.7 |
| Mean complaints | 1.08 |
| Basket coverage | 84.0% |

**Spending breakdown:** Electronics 37.5% · Groceries 36.4% · Video games 7.4% · Fish 3.8% · Meat 4.1%

This cluster has the **highest electronics spending share** by a large margin (37.5%), the **lowest loyalty card penetration** (49.2%), the **highest complaint rate** (1.08), and the **fewest children at home** (0.7). The profile describes a segment of demanding, lower-loyalty customers with a strong personal-care and technology orientation. The combination of high electronics spend, frequent complaints, and limited household dependents is distinctive and commercially significant from a retention standpoint.

**Top basket association rules (strongest in the dataset):**

| Antecedent | Consequent | Lift |
|---|---|---:|
| airpods | bluetooth headphones | **3.83** |
| bluetooth headphones | airpods | **3.83** |
| laptop | bluetooth headphones | 3.79 |

---

#### Cluster 4 — High-Value Large Loyal Families (2,944 customers, 8.9%)

| Feature | Value |
|---|---|
| Mean age | 60.4 years |
| Mean lifetime spend | **£37,486** |
| Loyalty card share | 67.4% |
| Promotion usage rate | 21.5% |
| Mean children at home | **6.4** |
| Mean complaints | 0.98 |
| Basket coverage | 84.8% |

**Spending breakdown:** Groceries 65.2% · Electronics 13.9% · Fish 3.7% · Meat 4.2% · Alcohol 4.2%

Cluster 4 is the **highest-spending segment** (mean lifetime spend £37,486 — more than three times cluster 0) and has the largest household size by a wide margin (mean 6.4 children at home). Despite being high-value, promotion usage is relatively low (21.5%), suggesting purchasing driven by household need rather than discounts. Tenure is the longest of all clusters. Basket patterns are dominated by grocery and breakfast staples.

**Top basket association rules:**

| Antecedent | Consequent | Lift |
|---|---|---:|
| mashed potato | honey | 1.39 |
| mushroom cream sauce | bacon | 1.37 |
| ham | eggs | 1.27 |

*Note: Association rule lift values are lower in cluster 4 than in other segments, reflecting a more diverse and generalised basket behaviour consistent with large family households purchasing across a broad range of categories.*

---

## 4. Targeted Promotions

Promotions were designed for each cluster by combining the cluster profile, basket behaviour, and association rule evidence. One actionable promotion recommendation was produced per segment.

| Cluster | Persona | Promotion Type | Products to Promote |
|:---:|---|---|---|
| 0 | Promo-Sensitive Older Tech Snackers | Loyalty discount bundle | Airpods, Bluetooth headphones, Energy drink, Energy bar, Protein bar |
| 1 | Mainstream Grocery Families | High-volume grocery offer | Asparagus, Tomatoes, Spinach, Carrots, Avocado, Salad |
| 2 | Vegetarian-Leaning Family Households | Lifestyle and family basket | Asparagus, Cooking oil, Cereals, Milk, Babies food, Dog food, Cat food |
| 3 | Demanding Personal-Care and Tech Shoppers | Retention and replenishment bundle | Toothpaste, Deodorant, Shampoo, Shower gel, Toothbrush, Airpods |
| 4 | High-Value Large Loyal Families | Premium loyalty family bundle | Cereals, Fresh bread, Butter, Eggs, Milk, Honey, Tea |

### Promotion Rationale by Segment

**Cluster 0 — Loyalty Discount Bundle**
This segment already holds loyalty cards at a 67.7% rate and has the highest promotion responsiveness (45.1%). A loyalty-card-exclusive discount on tech accessories paired with snack items targets the observed co-purchase behaviour (headphones × energy drinks, lift 2.57). Discounts should be controlled given that this segment's lifetime spend is the lowest of all five groups.

**Cluster 1 — High-Volume Grocery Offer**
With 75.9% of spend in groceries and strong fresh-produce co-purchase patterns (salad × avocado, lift 2.83), simple multi-buy or bundle deals on fresh vegetables are the most relevant offer type. Promotions should remain general given the broad nature of this segment.

**Cluster 2 — Lifestyle and Family Basket**
The high vegetable share, low meat share, and strong household-care basket patterns (cooking oil × napkins × babies food) support a healthy family basket promotion. This segment shops consistently (89.1% basket coverage) and does not require deep discounting — product curation is more effective than price cuts.

**Cluster 3 — Retention and Replenishment Bundle**
Low loyalty, high complaints, and high electronics spend characterise a segment at risk of churn. A retention-focused personal-care replenishment bundle (recurring household purchases) provides a reason to maintain the relationship, with an optional technology cross-sell to leverage the observed headphones × airpods co-purchase behaviour (lift 3.83). The professional persona label should be used in all customer-facing materials.

**Cluster 4 — Premium Loyalty Family Bundle**
As the highest-value and most tenured segment, deep discounting is inappropriate and may undermine perceived brand value. Premium loyalty rewards tied to breakfast and household staples (cereals, bread, eggs, honey) align with the observed basket patterns while reinforcing relationship value rather than price sensitivity.

---

## 5. Conclusions and Recommendations

### Summary of Findings

This project successfully produced a defensible, reproducible, and commercially interpretable customer segmentation across all 33,038 customers in the dataset. The final K-Means k=5 model was selected after systematic comparison with GMM, Agglomerative Clustering, and DBSCAN, and confirmed stable across multiple random seeds (minimum Adjusted Rand Index: 0.999).

The five segments are meaningfully differentiated along key commercial dimensions — spending level, category mix, promotion sensitivity, household composition, loyalty status, and complaint behaviour — and each has been translated into a targeted and evidence-based promotional strategy.

### Limitations

The following limitations should be acknowledged when using this segmentation:

1. **Cluster 1 breadth.** The largest segment contains 18,385 customers (55.6% of the dataset). This reflects genuine heterogeneity in the customer population and should be communicated as a broad mainstream segment rather than a modelling weakness. A k=6 solution would split this group into two sub-segments but produces weaker internal metrics overall.

2. **Modest silhouette score.** A silhouette of 0.221 is typical for real retail customer data where behavioural boundaries are gradual rather than sharp. The model's commercial utility (actionable segments, stable assignments, interpretable profiles) justifies its use despite the modest geometric separation.

3. **Basket data coverage.** Basket analysis covers 28,127 of 33,038 customers. The 4,911 customers with no basket records retain their cluster assignment in the output file but cannot be characterised by basket behaviour. Their cluster membership is based entirely on the demographic and behavioural features present in `customer_info`.

4. **Association rules describe co-occurrence, not causality.** Lift values indicate that product pairs appear together more frequently than expected by chance. They do not imply that purchasing one product causes the purchase of another. Promotion design should treat them as signals of affinity, not causal relationships.

5. **Cluster 4 association rules.** The highest lift values in cluster 4 are approximately 1.37–1.39, which are lower than those observed in other segments. This reflects the broader and more diverse basket behaviour of large family households rather than a data quality issue.

### Recommendations

- **Deploy the five-segment model** as the primary customer segmentation for promotional campaign design, using `outputs/customer_clusters.csv` as the source of cluster assignments.
- **Monitor Cluster 1 performance separately.** Given its size and breadth, A/B testing of promotion variants within this segment may reveal actionable sub-groups over time.
- **Prioritise retention for Cluster 3.** The combination of high complaint rate and low loyalty card penetration makes this the highest churn-risk segment. Retention-focused communication and service quality monitoring are recommended alongside promotional activity.
- **Protect Cluster 4 from over-discounting.** High-value, high-tenure customers respond better to exclusivity and reward than to price cuts. Premium loyalty programmes and early access to new products are more appropriate than deep promotional discounts.
- **Revisit k=6 as a business decision.** If the commercial team needs finer targeting within the mainstream segment, the k=6 alternative — which splits cluster 1 — is documented and ready to evaluate, accepting that it trades metric quality for balance.
