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


def format_percentage(value):
    return f"{value:.1%}"


def format_number(value):
    return f"{value:,.0f}"


def get_cluster_row(frame, cluster):
    matching_rows = frame.loc[frame["cluster"] == cluster]
    if matching_rows.empty:
        raise ValueError(f"Cluster {cluster} was not found.")

    return matching_rows.iloc[0]


def list_top_products(top_products, cluster, top_n=5, preferred_terms=None):
    cluster_products = (
        top_products.loc[top_products["cluster"] == cluster]
        .sort_values("product_rank")
    )

    if preferred_terms:
        preferred_terms = [term.lower() for term in preferred_terms]
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
        preferred_terms = [term.lower() for term in preferred_terms]
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


def create_cluster_promotion_recommendations(
    cluster_profile,
    spending_shares,
    basket_profile,
    top_products,
    association_rules,
):
    promotion_plan = {
        0: {
            "persona_name": "Promo-Sensitive Older Tech Snackers",
            "persona_summary": (
                "Older, loyal customers with high promotion usage and visible "
                "technology and snack signals."
            ),
            "recommended_promotion": (
                "Offer a tech-accessory bundle paired with energy drinks or snack bars, "
                "with a loyalty-card discount."
            ),
            "promotion_type": "Loyalty discount bundle",
            "products_to_promote": (
                "airpods, bluetooth headphones, energy drink, energy bar, protein bar"
            ),
            "risk_or_caveat": (
                "Spend is lower than other mature segments, so keep discounts controlled."
            ),
            "product_terms": [
                "airpods",
                "bluetooth headphones",
                "energy drink",
                "energy bar",
                "protein bar",
                "laptop",
            ],
            "rule_terms": ["airpods", "bluetooth headphones", "energy drink"],
        },
        1: {
            "persona_name": "Mainstream Grocery Families",
            "persona_summary": (
                "The broad main segment, dominated by grocery spend and family-oriented "
                "fresh-food baskets."
            ),
            "recommended_promotion": (
                "Use simple grocery and fresh-produce multi-buy offers for repeat basket "
                "items."
            ),
            "promotion_type": "High-volume grocery offer",
            "products_to_promote": (
                "asparagus, tomatoes, spinach, carrots, avocado, salad"
            ),
            "risk_or_caveat": (
                "This is the largest and broadest cluster, so promotions should stay general."
            ),
            "product_terms": [
                "asparagus",
                "tomatoes",
                "spinach",
                "carrots",
                "avocado",
                "salad",
            ],
            "rule_terms": ["salad", "avocado", "spinach", "carrots", "tomatoes"],
        },
        2: {
            "persona_name": "Vegetarian-Leaning Family Households",
            "persona_summary": (
                "Family households with the strongest vegetable share and very low meat "
                "and fish shares."
            ),
            "recommended_promotion": (
                "Create a healthy family basket around vegetables, cooking basics, baby "
                "items, and pet essentials."
            ),
            "promotion_type": "Lifestyle and family basket",
            "products_to_promote": (
                "asparagus, cooking oil, cereals, milk, babies food, dog food, cat food"
            ),
            "risk_or_caveat": (
                "This is vegetarian-leaning, not strictly vegetarian, because some meat "
                "products still appear."
            ),
            "product_terms": [
                "asparagus",
                "cooking oil",
                "cereals",
                "milk",
                "babies food",
                "dog food",
                "cat food",
                "napkins",
            ],
            "rule_terms": ["cooking oil", "napkins", "babies food", "dog food"],
        },
        3: {
            "persona_name": "Demanding Personal-Care and Tech Shoppers",
            "persona_summary": (
                "Lower-loyalty customers with the highest complaints, strong personal-care "
                "baskets, and the highest electronics share."
            ),
            "recommended_promotion": (
                "Use a retention-focused personal-care replenishment bundle with optional "
                "tech-accessory cross-sell."
            ),
            "promotion_type": "Retention and replenishment bundle",
            "products_to_promote": (
                "toothpaste, deodorant, shampoo, shower gel, tooth brush, airpods"
            ),
            "risk_or_caveat": (
                "Use a professional label in the report; the informal Karen-like idea is "
                "only an interpretation of complaints and product behaviour."
            ),
            "product_terms": [
                "toothpaste",
                "deodorant",
                "shampoo",
                "shower gel",
                "tooth brush",
                "airpods",
            ],
            "rule_terms": ["airpods", "bluetooth headphones", "laptop", "energy drink"],
        },
        4: {
            "persona_name": "High-Value Large Loyal Families",
            "persona_summary": (
                "The highest-spending and most tenured family segment, with very large "
                "households and breakfast/grocery baskets."
            ),
            "recommended_promotion": (
                "Offer premium loyalty rewards and family breakfast bundles rather than "
                "deep discounts."
            ),
            "promotion_type": "Premium loyalty family bundle",
            "products_to_promote": (
                "cereals, fresh bread, butter, eggs, milk, honey, tea"
            ),
            "risk_or_caveat": (
                "High-value customers may not need heavy discounts; focus on retention and value."
            ),
            "product_terms": [
                "cereals",
                "fresh bread",
                "butter",
                "eggs",
                "milk",
                "honey",
                "tea",
                "bacon",
            ],
            "rule_terms": ["cereals", "fresh bread", "butter", "eggs", "honey", "bacon"],
        },
    }

    recommendation_rows = []
    for cluster in sorted(promotion_plan):
        cluster_metrics = get_cluster_row(cluster_profile, cluster)
        share_metrics = get_cluster_row(spending_shares, cluster)
        basket_metrics = get_cluster_row(basket_profile, cluster)
        plan = promotion_plan[cluster]

        evidence_summary = (
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

        recommendation_rows.append(
            {
                "cluster": cluster,
                "persona_name": plan["persona_name"],
                "persona_summary": plan["persona_summary"],
                "main_evidence": evidence_summary,
                "recommended_promotion": plan["recommended_promotion"],
                "promotion_type": plan["promotion_type"],
                "products_to_promote": plan["products_to_promote"],
                "supporting_products": list_top_products(
                    top_products,
                    cluster,
                    top_n=5,
                    preferred_terms=plan["product_terms"],
                ),
                "supporting_association_rules": list_top_rules(
                    association_rules,
                    cluster,
                    top_n=2,
                    preferred_terms=plan["rule_terms"],
                ),
                "risk_or_caveat": plan["risk_or_caveat"],
            }
        )

    return pd.DataFrame(recommendation_rows)
