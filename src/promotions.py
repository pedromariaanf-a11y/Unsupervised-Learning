import ast
from collections import Counter
from itertools import combinations

import pandas as pd


def validate_customer_clusters(customer_clusters, expected_rows=33038):
    expected_columns = ["customer_id", "cluster"]
    if customer_clusters.columns.tolist() != expected_columns:
        raise ValueError(
            f"customer_clusters must have columns {expected_columns}, "
            f"found {customer_clusters.columns.tolist()}."
        )

    if len(customer_clusters) != expected_rows:
        raise ValueError(
            f"customer_clusters must have {expected_rows:,} rows, "
            f"found {len(customer_clusters):,}."
        )

    duplicated_customers = customer_clusters["customer_id"].duplicated().sum()
    if duplicated_customers > 0:
        raise ValueError(
            f"customer_clusters has {duplicated_customers:,} duplicated customer_id values."
        )

    missing_clusters = customer_clusters["cluster"].isna().sum()
    if missing_clusters > 0:
        raise ValueError(
            f"customer_clusters has {missing_clusters:,} missing cluster values."
        )


def parse_goods_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, list):
        return parsed

    return []


def add_basket_level_features(customer_basket):
    baskets = customer_basket.copy()
    baskets["goods_list"] = baskets["list_of_goods"].apply(parse_goods_list)
    baskets["basket_size"] = baskets["goods_list"].str.len()
    baskets["distinct_products_per_basket"] = baskets["goods_list"].apply(
        lambda goods: len(set(goods))
    )

    return baskets


def join_baskets_to_clusters(customer_basket, customer_clusters):
    validate_customer_clusters(customer_clusters)

    basket_clusters = customer_basket.merge(
        customer_clusters,
        on="customer_id",
        how="left",
    )

    missing_cluster_rows = basket_clusters["cluster"].isna().sum()
    if missing_cluster_rows > 0:
        raise ValueError(
            f"{missing_cluster_rows:,} basket rows did not match a final cluster."
        )

    return basket_clusters


def create_cluster_basket_profile(basket_clusters, customer_clusters):
    cluster_sizes = (
        customer_clusters.groupby("cluster")["customer_id"]
        .nunique()
        .reset_index(name="number_of_cluster_customers")
    )

    basket_profile = (
        basket_clusters.groupby("cluster")
        .agg(
            number_of_baskets=("invoice_id", "count"),
            number_of_customers_with_baskets=("customer_id", "nunique"),
            average_basket_size=("basket_size", "mean"),
            median_basket_size=("basket_size", "median"),
            average_distinct_products_per_basket=(
                "distinct_products_per_basket",
                "mean",
            ),
            total_items_in_baskets=("basket_size", "sum"),
        )
        .reset_index()
    )

    exploded_products = (
        basket_clusters[["cluster", "goods_list"]]
        .explode("goods_list")
        .dropna(subset=["goods_list"])
    )

    distinct_products = (
        exploded_products.groupby("cluster")["goods_list"]
        .nunique()
        .reset_index(name="number_of_distinct_products")
    )

    basket_profile = cluster_sizes.merge(basket_profile, on="cluster", how="left")
    basket_profile = basket_profile.merge(
        distinct_products,
        on="cluster",
        how="left",
    )

    fill_zero_columns = [
        "number_of_baskets",
        "number_of_customers_with_baskets",
        "average_basket_size",
        "median_basket_size",
        "average_distinct_products_per_basket",
        "total_items_in_baskets",
        "number_of_distinct_products",
    ]
    basket_profile[fill_zero_columns] = basket_profile[fill_zero_columns].fillna(0)

    basket_profile["share_of_cluster_customers_with_baskets"] = (
        basket_profile["number_of_customers_with_baskets"]
        / basket_profile["number_of_cluster_customers"]
    )

    return basket_profile[
        [
            "cluster",
            "number_of_baskets",
            "number_of_customers_with_baskets",
            "average_basket_size",
            "median_basket_size",
            "average_distinct_products_per_basket",
            "total_items_in_baskets",
            "number_of_distinct_products",
            "share_of_cluster_customers_with_baskets",
        ]
    ]


def create_cluster_top_products(basket_clusters, top_n=10):
    exploded_products = (
        basket_clusters[["cluster", "goods_list"]]
        .explode("goods_list")
        .dropna(subset=["goods_list"])
        .rename(columns={"goods_list": "product"})
    )

    product_counts = (
        exploded_products.groupby(["cluster", "product"])
        .size()
        .reset_index(name="product_count")
    )

    product_counts["product_share_within_cluster"] = (
        product_counts["product_count"]
        / product_counts.groupby("cluster")["product_count"].transform("sum")
    )

    product_counts = product_counts.sort_values(
        ["cluster", "product_count", "product"],
        ascending=[True, False, True],
    )
    product_counts["product_rank"] = (
        product_counts.groupby("cluster").cumcount() + 1
    )

    top_products = product_counts[product_counts["product_rank"] <= top_n].copy()

    return top_products[
        [
            "cluster",
            "product",
            "product_count",
            "product_share_within_cluster",
            "product_rank",
        ]
    ]


def format_itemset(itemset):
    return " + ".join(sorted(itemset))


def empty_association_rules_frame():
    return pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "antecedent_support",
            "consequent_support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
        ]
    )


def mine_pairwise_association_rules(
    basket_lists,
    min_support=0.01,
    min_confidence=0.15,
    min_lift=1.2,
):
    transactions = [sorted(set(basket)) for basket in basket_lists if len(basket) > 0]
    number_of_transactions = len(transactions)
    if number_of_transactions == 0:
        return empty_association_rules_frame()

    item_counts = Counter()
    pair_counts = Counter()
    for transaction in transactions:
        item_counts.update(transaction)
        pair_counts.update(combinations(transaction, 2))

    rule_rows = []
    for (left_product, right_product), pair_count in pair_counts.items():
        support = pair_count / number_of_transactions
        if support < min_support:
            continue

        for antecedent, consequent in [
            (left_product, right_product),
            (right_product, left_product),
        ]:
            antecedent_support = item_counts[antecedent] / number_of_transactions
            consequent_support = item_counts[consequent] / number_of_transactions
            confidence = pair_count / item_counts[antecedent]
            lift = confidence / consequent_support

            if confidence < min_confidence or lift < min_lift:
                continue

            leverage = support - (antecedent_support * consequent_support)
            conviction = (
                float("inf")
                if confidence == 1
                else (1 - consequent_support) / (1 - confidence)
            )

            rule_rows.append(
                {
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "antecedent_support": antecedent_support,
                    "consequent_support": consequent_support,
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                }
            )

    if not rule_rows:
        return empty_association_rules_frame()

    return pd.DataFrame(rule_rows)


def mine_association_rules(
    basket_lists,
    min_support=0.01,
    min_confidence=0.15,
    min_lift=1.2,
    max_len=2,
):
    if max_len != 2:
        raise ValueError("Only pairwise association rules are supported in this phase.")

    transactions = [basket for basket in basket_lists if len(basket) > 0]
    if not transactions:
        return empty_association_rules_frame()

    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
        from mlxtend.preprocessing import TransactionEncoder

        encoder = TransactionEncoder()
        encoded_array = encoder.fit(transactions).transform(transactions)
        encoded_baskets = pd.DataFrame(encoded_array, columns=encoder.columns_)

        frequent_itemsets = fpgrowth(
            encoded_baskets,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
        if frequent_itemsets.empty:
            return empty_association_rules_frame()

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        if rules.empty:
            return empty_association_rules_frame()

        rules = rules[rules["lift"] >= min_lift].copy()
        if rules.empty:
            return empty_association_rules_frame()

        rules["antecedents"] = rules["antecedents"].apply(format_itemset)
        rules["consequents"] = rules["consequents"].apply(format_itemset)
        rules = rules.rename(
            columns={
                "antecedent support": "antecedent_support",
                "consequent support": "consequent_support",
            }
        )

        return rules
    except Exception:
        return mine_pairwise_association_rules(
            transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
        )


def create_cluster_association_rules(
    basket_clusters,
    min_support=0.01,
    min_confidence=0.15,
    min_lift=1.2,
    max_len=2,
    top_n=20,
):
    rule_frames = []

    for cluster in sorted(basket_clusters["cluster"].dropna().unique()):
        cluster_baskets = basket_clusters.loc[
            basket_clusters["cluster"] == cluster,
            "goods_list",
        ].tolist()

        rules = mine_association_rules(
            cluster_baskets,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
        )
        if rules.empty:
            continue

        rules.insert(0, "cluster", int(cluster))
        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=[False, False, False],
        ).head(top_n)
        rules["rule_rank"] = range(1, len(rules) + 1)
        rule_frames.append(rules)

    if not rule_frames:
        return pd.DataFrame(
            columns=[
                "cluster",
                "antecedents",
                "consequents",
                "antecedent_support",
                "consequent_support",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
                "rule_rank",
            ]
        )

    cluster_rules = pd.concat(rule_frames, ignore_index=True)

    output_columns = [
        "cluster",
        "antecedents",
        "consequents",
        "antecedent_support",
        "consequent_support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "rule_rank",
    ]
    output_columns = [
        column for column in output_columns
        if column in cluster_rules.columns
    ]
    cluster_rules = cluster_rules[output_columns]

    return cluster_rules
