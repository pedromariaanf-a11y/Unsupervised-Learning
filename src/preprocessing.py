import pandas as pd
from sklearn.preprocessing import StandardScaler


NON_MODEL_COLUMNS = ["most_frequent_product"]
CATEGORICAL_COLUMNS = ["customer_gender", "degree_level"]


def clean_feature_values(df):
    cleaned = df.copy()

    if "percentage_of_products_bought_promotion" in cleaned.columns:
        cleaned["percentage_of_products_bought_promotion"] = cleaned[
            "percentage_of_products_bought_promotion"
        ].clip(lower=0, upper=100)

    numeric_columns = cleaned.select_dtypes(include="number").columns
    for column in numeric_columns:
        median_value = cleaned[column].median()
        if pd.isna(median_value):
            median_value = 0
        cleaned[column] = cleaned[column].fillna(median_value)

    categorical_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in categorical_columns:
        cleaned[column] = cleaned[column].fillna("Unknown")

    return cleaned


def select_model_features(df):
    selected = df.copy()
    columns_to_drop = [column for column in NON_MODEL_COLUMNS if column in selected.columns]
    return selected.drop(columns=columns_to_drop)


def preprocess_for_clustering(df):
    cleaned = clean_feature_values(df)
    selected = select_model_features(cleaned)

    customer_ids = selected["customer_id"].copy()
    model_features = selected.drop(columns=["customer_id"])

    categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in model_features.columns
    ]
    numeric_columns = model_features.drop(columns=categorical_columns).select_dtypes(
        include="number"
    ).columns

    encoded_features = pd.get_dummies(
        model_features,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    scaler = StandardScaler()
    encoded_features[numeric_columns] = scaler.fit_transform(
        encoded_features[numeric_columns]
    )

    preprocessed = encoded_features.copy()
    preprocessed.insert(0, "customer_id", customer_ids.values)

    return preprocessed
