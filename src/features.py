import ast

import numpy as np
import pandas as pd


def get_lifetime_spend_columns(df):
    return [column for column in df.columns if column.startswith("lifetime_spend_")]


def extract_degree_level(customer_name):
    if pd.isna(customer_name):
        return "Unknown"

    name = str(customer_name).strip()
    if name.startswith("Bsc."):
        return "Bsc"
    if name.startswith("Msc."):
        return "Msc"
    if name.startswith("Phd."):
        return "Phd"

    return "Unknown"


def add_customer_info_features(df):
    features = df.copy()

    transaction_year = pd.to_numeric(features["year_first_transaction"], errors="coerce")
    plausible_transaction_years = transaction_year[transaction_year <= 2026]
    reference_year = int(plausible_transaction_years.max())

    birthdate = pd.to_datetime(
        features["customer_birthdate"],
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )

    features["age"] = reference_year - birthdate.dt.year
    features["customer_tenure"] = reference_year - transaction_year
    features["has_loyalty_card"] = features["loyalty_card_number"].notna().astype(int)
    features["total_children_home"] = features["kids_home"] + features["teens_home"]
    features["has_children"] = (features["total_children_home"].fillna(0) > 0).astype(int)

    spend_columns = get_lifetime_spend_columns(features)
    features["total_lifetime_spend"] = features[spend_columns].sum(axis=1, min_count=1)

    for column in spend_columns:
        share_column = f"share_{column.replace('lifetime_spend_', '')}"
        features[share_column] = np.where(
            features["total_lifetime_spend"] > 0,
            features[column] / features["total_lifetime_spend"],
            0,
        )

    features["degree_level"] = features["customer_name"].apply(extract_degree_level)

    original_columns = [
        "customer_id",
        "customer_gender",
        "kids_home",
        "teens_home",
        "number_complaints",
        "distinct_stores_visited",
        "lifetime_total_distinct_products",
        "year_first_transaction",
        "percentage_of_products_bought_promotion",
        "typical_hour",
        "latitude",
        "longitude",
    ]
    share_columns = [f"share_{column.replace('lifetime_spend_', '')}" for column in spend_columns]
    engineered_columns = [
        "age",
        "customer_tenure",
        "has_loyalty_card",
        "total_children_home",
        "has_children",
        "total_lifetime_spend",
        *share_columns,
        "degree_level",
    ]

    return features[original_columns + engineered_columns]


def parse_goods_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    parsed_value = ast.literal_eval(value)
    if isinstance(parsed_value, list):
        return parsed_value

    return []


def add_basket_size_columns(df):
    baskets = df.copy()
    baskets["goods_list"] = baskets["list_of_goods"].apply(parse_goods_list)
    baskets["basket_size"] = baskets["goods_list"].str.len()
    baskets["distinct_products_per_basket"] = baskets["goods_list"].apply(
        lambda goods: len(set(goods))
    )

    return baskets


def create_basket_features(customer_basket):
    baskets = add_basket_size_columns(customer_basket)

    basket_features = (
        baskets.groupby("customer_id")
        .agg(
            basket_count=("invoice_id", "count"),
            avg_basket_size=("basket_size", "mean"),
            max_basket_size=("basket_size", "max"),
            min_basket_size=("basket_size", "min"),
            total_items_bought_in_baskets=("basket_size", "sum"),
            avg_distinct_products_per_basket=("distinct_products_per_basket", "mean"),
        )
        .reset_index()
    )

    exploded_products = (
        baskets[["customer_id", "goods_list"]]
        .explode("goods_list")
        .dropna(subset=["goods_list"])
    )

    distinct_products = (
        exploded_products.groupby("customer_id")["goods_list"]
        .nunique()
        .reset_index(name="distinct_products_in_baskets")
    )

    product_counts = (
        exploded_products.groupby(["customer_id", "goods_list"])
        .size()
        .reset_index(name="product_count")
        .sort_values(
            ["customer_id", "product_count", "goods_list"],
            ascending=[True, False, True],
        )
    )
    most_frequent_product = (
        product_counts.drop_duplicates("customer_id")[
            ["customer_id", "goods_list"]
        ].rename(columns={"goods_list": "most_frequent_product"})
    )

    basket_features = basket_features.merge(
        distinct_products, on="customer_id", how="left"
    )
    basket_features = basket_features.merge(
        most_frequent_product, on="customer_id", how="left"
    )

    return basket_features[
        [
            "customer_id",
            "basket_count",
            "avg_basket_size",
            "max_basket_size",
            "min_basket_size",
            "total_items_bought_in_baskets",
            "distinct_products_in_baskets",
            "avg_distinct_products_per_basket",
            "most_frequent_product",
        ]
    ]
