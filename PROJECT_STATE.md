# Project State

- Project type: customer segmentation project.
- Current state: model comparison completed; K-Means k=5 is the strongest practical candidate so far; project is ready to confirm the final cluster output and move to basket-based profiling, association rules, and targeted promotions.
- Completed work: K-Means baseline with k=5; initial cluster profiling; PCA and t-SNE visualizations; K-Means vs GMM comparison; Agglomerative comparison on a reproducible sample; DBSCAN outlier diagnostic on the same sample; reusable clustering helper functions reviewed, with the baseline notebook now reusing the existing K-Means helper.
- Validated outputs: `outputs/clustering/kmeans_customer_clusters.csv` has 33,038 rows, no duplicated `customer_id` values, no missing `cluster` values, and exactly `customer_id` and `cluster` columns; K-Means metrics and profiling outputs exist; model comparison outputs exist; complementary comparison outputs exist; `data/processed/selected_model_features.csv` was not modified.
- Known decisions/risks: K-Means k=5 remains the strongest practical candidate so far; GMM did not improve the segmentation and produced one very large cluster; Agglomerative Clustering gave some support to a 5-cluster structure but did not outperform K-Means; DBSCAN is not suitable as the main segmentation model under the tested grid and should only be mentioned as an outlier diagnostic; final model choice still needs to be connected to business usefulness and promotion design.
- Recommended next task: confirm K-Means k=5 as the final model candidate, create `outputs/customer_clusters.csv`, and proceed to basket-based profiling and association rules by cluster.
- Blockers: none currently known.
