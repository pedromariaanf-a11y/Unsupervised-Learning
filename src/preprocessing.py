import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


NON_MODEL_COLUMNS = ["most_frequent_product"]
FINAL_MODEL_COLUMNS = [
    "number_complaints",
    "distinct_stores_visited",
    "lifetime_total_distinct_products",
    "percentage_of_products_bought_promotion",
    "typical_hour",
    "age",
    "customer_tenure",
    "has_loyalty_card",
    "total_children_home",
    "total_lifetime_spend",
    "share_groceries",
    "share_electronics",
    "share_vegetables",
    "share_nonalcohol_drinks",
    "share_alcohol_drinks",
    "share_meat",
    "share_fish",
    "share_hygiene",
    "share_videogames",
    "share_petfood",
]


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
    selected = selected.drop(columns=columns_to_drop)

    required_columns = ["customer_id", *FINAL_MODEL_COLUMNS]
    missing_columns = [
        column for column in required_columns
        if column not in selected.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {missing_columns}")

    return selected[required_columns]


def preprocess_for_clustering(df):
    cleaned = clean_feature_values(df)
    selected = select_model_features(cleaned)

    customer_ids = selected["customer_id"].copy()
    model_features = selected.drop(columns=["customer_id"])

    non_numeric_columns = model_features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Non-numeric model columns found: {non_numeric_columns}")

    scaler = RobustScaler()
    scaled_features = model_features.copy()
    scaled_features[FINAL_MODEL_COLUMNS] = scaler.fit_transform(
        scaled_features[FINAL_MODEL_COLUMNS]
    )

    preprocessed = scaled_features.copy()
    preprocessed.insert(0, "customer_id", customer_ids.values)

    return preprocessed


def scale_model_features(df, feature_columns=None, scaler=None, use_log=False):
    if feature_columns is None:
        feature_columns = FINAL_MODEL_COLUMNS
    if scaler is None:
        scaler = RobustScaler()

    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    features = df[feature_columns].copy()
    if use_log:
        features = np.log1p(features.clip(lower=0))

    scaled_features = pd.DataFrame(
        scaler.fit_transform(features),
        columns=feature_columns,
        index=df.index,
    )

    return scaled_features
