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


ASSOCIATION_RULE_COLUMNS = [
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

PROMOTION_PLAN_COLUMNS = [
    "cluster",
    "persona_name",
    "persona_summary",
    "recommended_promotion",
    "promotion_type",
    "products_to_promote",
    "risk_or_caveat",
]

PROMOTION_OUTPUT_COLUMNS = [
    "cluster",
    "persona_name",
    "persona_summary",
    "main_evidence",
    "recommended_promotion",
    "promotion_type",
    "products_to_promote",
    "supporting_products",
    "supporting_association_rules",
    "risk_or_caveat",
]


def empty_association_rules_frame():
    return pd.DataFrame(columns=ASSOCIATION_RULE_COLUMNS)


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

    return pd.DataFrame(rule_rows, columns=ASSOCIATION_RULE_COLUMNS)


def create_cluster_association_rules(
    basket_clusters,
    min_support=0.01,
    min_confidence=0.15,
    min_lift=1.2,
    top_n=20,
):
    rule_frames = []

    for cluster in sorted(basket_clusters["cluster"].dropna().unique()):
        cluster_baskets = basket_clusters.loc[
            basket_clusters["cluster"] == cluster,
            "goods_list",
        ].tolist()

        rules = mine_pairwise_association_rules(
            cluster_baskets,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
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
        *ASSOCIATION_RULE_COLUMNS,
        "rule_rank",
    ]

    return cluster_rules[output_columns]


def format_percentage(value):
    return f"{value:.1%}"


def format_number(value):
    return f"{value:,.0f}"


def get_cluster_row(frame, cluster):
    matching_rows = frame.loc[frame["cluster"] == cluster]
    if matching_rows.empty:
        raise ValueError(f"Cluster {cluster} was not found.")

    return matching_rows.iloc[0]


def normalize_terms(terms):
    if terms is None:
        return []
    if isinstance(terms, str):
        return [term.strip().lower() for term in terms.split(",") if term.strip()]

    return [str(term).lower() for term in terms]


def list_top_products(top_products, cluster, top_n=5, preferred_terms=None):
    cluster_products = (
        top_products.loc[top_products["cluster"] == cluster]
        .sort_values("product_rank")
    )

    if preferred_terms:
        preferred_terms = normalize_terms(preferred_terms)
        product_text = cluster_products["product"].astype(str).str.lower()
        relevant_products = cluster_products.loc[
            product_text.apply(
                lambda text: any(term in text for term in preferred_terms)
            )
        ]
        if not relevant_products.empty:
            cluster_products = relevant_products

    products = cluster_products.head(top_n)["product"].tolist()

    return ", ".join(products)


def list_top_rules(association_rules, cluster, top_n=2, preferred_terms=None):
    rules = (
        association_rules.loc[association_rules["cluster"] == cluster]
        .sort_values("rule_rank")
    )

    if preferred_terms:
        preferred_terms = normalize_terms(preferred_terms)
        rule_text = (
            rules["antecedents"].astype(str)
            + " "
            + rules["consequents"].astype(str)
        ).str.lower()
        relevant_rules = rules.loc[
            rule_text.apply(
                lambda text: any(term in text for term in preferred_terms)
            )
        ]
        if not relevant_rules.empty:
            rules = relevant_rules

    rules = rules.head(top_n)
    if rules.empty:
        return "No strong pairwise rules saved."

    return "; ".join(
        f"{row.antecedents} -> {row.consequents} (lift {row.lift:.2f})"
        for row in rules.itertuples(index=False)
    )


def validate_promotion_plan(promotion_plan):
    missing_columns = [
        column for column in PROMOTION_PLAN_COLUMNS
        if column not in promotion_plan.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing promotion plan columns: {missing_columns}")


def create_cluster_evidence(cluster, cluster_profile, spending_shares, basket_profile):
    cluster_metrics = get_cluster_row(cluster_profile, cluster)
    share_metrics = get_cluster_row(spending_shares, cluster)
    basket_metrics = get_cluster_row(basket_profile, cluster)

    return (
        f"Mean age {cluster_metrics.age_mean:.1f}; "
        f"mean spend {format_number(cluster_metrics.total_lifetime_spend_mean)}; "
        f"loyalty-card share {format_percentage(cluster_metrics.has_loyalty_card_mean)}; "
        f"children at home {cluster_metrics.total_children_home_mean:.1f}; "
        f"groceries {format_percentage(share_metrics.share_groceries)}, "
        f"electronics {format_percentage(share_metrics.share_electronics)}, "
        f"vegetables {format_percentage(share_metrics.share_vegetables)}, "
        f"meat {format_percentage(share_metrics.share_meat)}, "
        f"basket coverage "
        f"{format_percentage(basket_metrics.share_of_cluster_customers_with_baskets)}."
    )


def get_optional_terms(row, column):
    if column not in row.index:
        return None

    value = row[column]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    return value


def create_cluster_promotion_recommendations(
    promotion_plan,
    cluster_profile,
    spending_shares,
    basket_profile,
    top_products,
    association_rules,
):
    validate_promotion_plan(promotion_plan)

    recommendation_rows = []
    for _, plan in promotion_plan.sort_values("cluster").iterrows():
        cluster = int(plan["cluster"])

        recommendation_rows.append(
            {
                "cluster": cluster,
                "persona_name": plan["persona_name"],
                "persona_summary": plan["persona_summary"],
                "main_evidence": create_cluster_evidence(
                    cluster,
                    cluster_profile,
                    spending_shares,
                    basket_profile,
                ),
                "recommended_promotion": plan["recommended_promotion"],
                "promotion_type": plan["promotion_type"],
                "products_to_promote": plan["products_to_promote"],
                "supporting_products": list_top_products(
                    top_products,
                    cluster,
                    top_n=5,
                    preferred_terms=get_optional_terms(plan, "product_terms"),
                ),
                "supporting_association_rules": list_top_rules(
                    association_rules,
                    cluster,
                    top_n=2,
                    preferred_terms=get_optional_terms(plan, "rule_terms"),
                ),
                "risk_or_caveat": plan["risk_or_caveat"],
            }
        )

    return pd.DataFrame(recommendation_rows, columns=PROMOTION_OUTPUT_COLUMNS)
