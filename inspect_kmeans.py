import joblib
import pandas as pd
import numpy as np

try:
    kmeans = joblib.load('models/kmeans_model.pkl')
    print("Cluster Centers (Scaled):")
    print(kmeans.cluster_centers_)
    
    scaler = joblib.load('models/scaler.pkl')
    unscaled_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    print("Cluster Centers (Unscaled - R, F, M):")
    print(unscaled_centers)
except Exception as e:
    print("Error:", e)
