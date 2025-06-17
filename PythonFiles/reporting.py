import pandas as pd
from scipy.stats import spearmanr

# depending path
with open("results_dict.pkl", "rb") as f:
    results_dict = pickle.load(f)

# 1) Collect predictive performance metrics into a DataFrame
metrics = []
for method, res in results_dict.items():
    # Adapt these keys to whatever metrics you logged:
    entry = {
        "method": method,
        "test_accuracy": res.get("test_accuracy", None),
        "test_roc_auc": res.get("test_roc_auc", res.get("test roc auc score", None)),
        "rmse_test":    res.get("rmse_test", None),
        "mae_test":     res.get("mae_test", None),
        "r2_test":      res.get("r2_test", None)
    }
    metrics.append(entry)
perf_df = pd.DataFrame(metrics).set_index("method")
print("=== Predictive Performance ===")
print(perf_df)

# 2) Aggregate feature importances across methods into a DataFrame
#    (only methods that stored "feature_importances" or "global_importance")
fi_dict = {}
for method, res in results_dict.items():
    if "feature_importances" in res:
        fi = res["feature_importances"]
        fi_dict[method] = fi
    elif "global_importance" in res:
        # EBM’s global_importance is a list of dicts: [{"feature": name, "score": s}, …]
        # Convert to a numeric array aligned by feature index:
        gi = pd.Series({d["feature"]: d["score"] for d in res["global_importance"]})
        # Sort by feature name to align across methods
        fi_dict[method] = gi.sort_index().values

# Build DataFrame: rows=features, cols=methods
# (assumes all arrays same length and same feature order; otherwise index with feature names)
fi_df = pd.DataFrame(fi_dict)
print("\n=== Feature Importances ===")
print(fi_df.head())

# 3) Compute pairwise Spearman rank correlations
corr = fi_df.corr(method="spearman")
print("\n=== Spearman Rank-Correlation of Feature Importances ===")
print(corr)

# 4) If you want a long-format table of correlations:
corr_long = corr.reset_index().melt(id_vars="index", var_name="method2", value_name="spearman_r")
corr_long.columns = ["method1", "method2", "spearman_r"]
print("\n=== Pairwise Spearman Correlations (long format) ===")
print(corr_long)
