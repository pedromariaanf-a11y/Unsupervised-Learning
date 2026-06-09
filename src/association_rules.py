import pandas as pd
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def extract_cluster_rules(basket_lists, min_support=0.02, min_confidence=0.1, min_lift=1.2):
    """
    Runs FP-Growth on a list of baskets (transactions) and returns filtered rules.
    """
    if len(basket_lists) == 0:
        return pd.DataFrame()

    # 1. One-Hot Encode the baskets
    te = TransactionEncoder()
    te_ary = te.fit(basket_lists).transform(basket_lists)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 2. Find frequent itemsets
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return pd.DataFrame()

    # 3. Generate rules based on confidence
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    # 4. Filter by lift and sort by highest impact
    filtered_rules = rules[rules['lift'] >= min_lift].sort_values(
        by=['lift', 'confidence'], ascending=[False, False]
    )

    return filtered_rules

def get_promotions_per_cluster(merged_df, cluster_col='cluster', basket_col='list_of_goods'):
    """
    Iterates over each cluster and extracts their specific association rules.
    """
    cluster_promotions = {}
    clusters = merged_df[cluster_col].dropna().unique()

    for c in sorted(clusters):
        print(f"Processing rules for Cluster {int(c)}...")
        
        # Filter baskets for this specific cluster
        cluster_data = merged_df[merged_df[cluster_col] == c]
        
        # Ensure basket data is in a clean list format
        baskets = []
        for item in cluster_data[basket_col]:
            if isinstance(item, str):
                try:
                    # Safely evaluate string representation of a list
                    baskets.append(ast.literal_eval(item))
                except (ValueError, SyntaxError):
                    continue
            elif isinstance(item, list):
                baskets.append(item)

        # Extract rules for this subset
        rules = extract_cluster_rules(baskets)
        cluster_promotions[c] = rules
        print(f"  -> Found {len(rules)} strong rules.")

    return cluster_promotions
