from itertools import combinations

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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


def sample_clustering_data(X, customer_ids=None, sample_size=5000, random_state=42):
    sample_size = min(sample_size, len(X))
    sample_indices = pd.Series(range(len(X))).sample(
        n=sample_size,
        random_state=random_state,
    ).tolist()

    X_sample = X.iloc[sample_indices].reset_index(drop=True)

    if customer_ids is None:
        return X_sample

    sampled_customer_ids = pd.Series(customer_ids).iloc[sample_indices].reset_index(drop=True)

    return X_sample, sampled_customer_ids


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


def fit_kmeans(X, n_clusters, random_state=42, n_init=50):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X)

    return model, labels


def evaluate_kmeans(X, k, random_state=42, n_init=50, sample_size=10000):
    model, labels = fit_kmeans(
        X,
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    )
    metrics = calculate_clustering_metrics(
        X,
        labels,
        sample_size=sample_size,
        random_state=random_state,
    )
    metrics.update({
        "k": k,
        "random_state": random_state,
        "n_features": X.shape[1],
        "inertia": model.inertia_,
    })

    return metrics, labels


def compare_kmeans_k_values(X, k_values, random_state=42, n_init=50, sample_size=10000):
    rows = []

    for k in k_values:
        metrics, _ = evaluate_kmeans(
            X,
            k=k,
            random_state=random_state,
            n_init=n_init,
            sample_size=sample_size,
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def compare_kmeans_feature_sets(
    feature_sets,
    k=5,
    random_state=42,
    n_init=50,
    sample_size=10000,
):
    rows = []

    for variant, features in feature_sets.items():
        metrics, _ = evaluate_kmeans(
            features,
            k=k,
            random_state=random_state,
            n_init=n_init,
            sample_size=sample_size,
        )
        metrics["feature_set_variant"] = variant
        rows.append(metrics)

    return pd.DataFrame(rows)


def calculate_seed_stability(X, k, seeds, n_init=50):
    labels_by_seed = {}

    for seed in seeds:
        _, labels = evaluate_kmeans(
            X,
            k=k,
            random_state=seed,
            n_init=n_init,
        )
        labels_by_seed[seed] = labels

    rows = []
    for seed_a, seed_b in combinations(seeds, 2):
        rows.append({
            "k": k,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "adjusted_rand_index": adjusted_rand_score(
                labels_by_seed[seed_a],
                labels_by_seed[seed_b],
            ),
        })

    return pd.DataFrame(rows)


def fit_gmm(X, n_components, covariance_type="diag", random_state=42):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    model.fit(X)
    labels = model.predict(X)

    return model, labels


def evaluate_clustering_solution(
    X,
    labels,
    model_name,
    algorithm,
    n_clusters,
    extra_metrics=None,
):
    metrics = calculate_clustering_metrics(X, labels)
    metrics.update({
        "model": model_name,
        "algorithm": algorithm,
        "n_clusters": n_clusters,
    })

    if extra_metrics is not None:
        metrics.update(extra_metrics)

    return metrics


def create_cluster_profile(df, cluster_column="cluster", columns=None):
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
        columns = [column for column in columns if column != cluster_column]
    else:
        columns = [
            column
            for column in columns
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
        ]

    profile = df.groupby(cluster_column)[columns].agg(["mean", "median"]).reset_index()
    profile.columns = [
        column if statistic == "" else f"{column}_{statistic}"
        for column, statistic in profile.columns
    ]

    return profile


def create_tsne_projection(X, labels, customer_ids=None, sample_size=5000, random_state=42):
    sample_size = min(sample_size, len(X))
    sample_indices = pd.Series(range(len(X))).sample(
        n=sample_size,
        random_state=random_state,
    ).tolist()

    X_sample = X.iloc[sample_indices]
    sampled_labels = pd.Series(labels).iloc[sample_indices].reset_index(drop=True)

    tsne = TSNE(n_components=2, random_state=random_state)
    projection = tsne.fit_transform(X_sample)

    tsne_df = pd.DataFrame({
        "tsne_1": projection[:, 0],
        "tsne_2": projection[:, 1],
        "cluster": sampled_labels,
    })

    if customer_ids is not None:
        tsne_df["customer_id"] = pd.Series(customer_ids).iloc[sample_indices].reset_index(drop=True)

    return tsne_df


def save_cluster_assignments(customer_ids, labels, output_path):
    assignments = pd.DataFrame({
        "customer_id": customer_ids,
        "cluster": labels,
    })

    assignments.to_csv(output_path, index=False)

    return assignments
