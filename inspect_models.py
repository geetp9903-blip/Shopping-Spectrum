import joblib
import pandas as pd

cluster_mapping = joblib.load('models/cluster_mapping.pkl')
print("Cluster mapping:")
print(cluster_mapping)

try:
    product_similarity = joblib.load('models/product_similarity.pkl')
    print("Product similarity type:", type(product_similarity))
    if isinstance(product_similarity, pd.DataFrame):
        print("Columns:", product_similarity.columns[:5])
        print("Shape:", product_similarity.shape)
    elif isinstance(product_similarity, dict):
        print("Keys snippet:", list(product_similarity.keys())[:5])
except Exception as e:
    print("Error loading product similarity:", e)
