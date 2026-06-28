import streamlit as st
import pandas as pd
import joblib
import os

# Define relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'online_retail.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_data
def load_data():
    """Load and preprocess the online retail dataset. Cached for performance."""
    try:
        # Load the dataset
        df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
        
        # Drop missing customer IDs
        df = df.dropna(subset=['CustomerID'])
        df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
        
        # Data types and derived columns
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Remove cancelled orders
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    """Load ML models and similarity matrices. Cached for performance."""
    try:
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
        cluster_mapping = joblib.load(os.path.join(MODELS_DIR, 'cluster_mapping.pkl'))
        product_similarity = joblib.load(os.path.join(MODELS_DIR, 'product_similarity.pkl'))
        return scaler, kmeans_model, cluster_mapping, product_similarity
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None
