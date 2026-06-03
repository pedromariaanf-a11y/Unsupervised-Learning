import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def validate_clustering_input(df, id_column="customer_id"):
    if id_column not in df.columns:
        raise ValueError(f"{id_column} column is missing from the clustering input.")

    if df[id_column].duplicated().sum() > 0:
        raise ValueError(f"Duplicated {id_column} values found in the clustering input.")

    if df.isna().sum().sum() > 0:
        raise ValueError("Missing values found in the clustering input.")

    X = df.drop(columns=[id_column])
    non_numeric_columns = X.select_dtypes(exclude="number").columns.tolist()

    if non_numeric_columns:
        raise ValueError(f"Non-numeric modelling columns found: {non_numeric_columns}")


def split_customer_features(df, id_column="customer_id"):
    customer_ids = df[id_column]
    X = df.drop(columns=[id_column])

    return customer_ids, X


def get_cluster_summary(labels):
    summary = pd.Series(labels).value_counts().sort_index().reset_index()
    summary.columns = ["cluster", "count"]
    summary["percentage"] = summary["count"] / len(labels) * 100

    return summary


def calculate_clustering_metrics(X, labels, sample_size=10000, random_state=42):
    sample_size = min(sample_size, len(X))
    cluster_summary = get_cluster_summary(labels)

    return {
        "silhouette_score": silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=random_state,
        ),
        "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        "davies_bouldin_score": davies_bouldin_score(X, labels),
        "min_cluster_size": int(cluster_summary["count"].min()),
        "max_cluster_size": int(cluster_summary["count"].max()),
        "min_cluster_percentage": cluster_summary["percentage"].min(),
        "max_cluster_percentage": cluster_summary["percentage"].max(),
    }


def save_cluster_assignments(customer_ids, labels, output_path):
    assignments = pd.DataFrame({
        "customer_id": customer_ids,
        "cluster": labels,
    })

    assignments.to_csv(output_path, index=False)

    return assignments
