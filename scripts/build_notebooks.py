import os
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Header
    cells.append(nbf.v4.new_markdown_cell(
        "# 📊 Shopper Spectrum: Exploratory Data Analysis (EDA)\n"
        "This notebook performs the Exploratory Data Analysis (EDA) for the **Shopper Spectrum** project. "
        "We will explore the dataset, clean the raw retail transaction data, and extract behavioral insights such as:\n"
        "* Transaction volumes and revenues across different countries\n"
        "* Top-selling products by quantity and revenue\n"
        "* Sales trends over time (monthly, daily, hourly)\n"
        "* Distributions of transactional attributes (Quantity, UnitPrice, and Order Value)"
    ))
    
    # 2. Imports
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from IPython.display import display\n"
        "\n"
        "# Set visualization parameters\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n"
        "plt.rcParams[\"font.size\"] = 12\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')"
    ))
    
    # 3. Loading Dataset
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load the Dataset\n"
        "We will load the raw dataset from `../data/online_retail.csv` and inspect its shape, columns, and data types."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Load data\n"
        "csv_path = '../data/online_retail.csv'\n"
        "df = pd.read_csv(csv_path)\n"
        "print(f\"Raw dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns.\\n\")\n"
        "df.info()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Inspect first 5 rows\n"
        "df.head()"
    ))
    
    # 4. Check missing values
    cells.append(nbf.v4.new_markdown_cell(
        "### Inspect Missing Values & Anomalies\n"
        "Let's see how many missing values exist in each column."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "missing_counts = df.isnull().sum()\n"
        "missing_percentage = (missing_counts / len(df)) * 100\n"
        "missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_percentage.round(2)})\n"
        "missing_df"
    ))
    
    # 5. Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Data Preprocessing & Cleaning\n"
        "To ensure the quality of our clustering and recommendation models, we apply the following cleaning steps:\n"
        "1. **Remove rows with missing CustomerID:** Since we need CustomerID to perform customer-level RFM analysis and segmentation.\n"
        "2. **Exclude cancelled invoices:** Invoices starting with 'C' represent cancellations/refunds.\n"
        "3. **Remove negative or zero quantities and unit prices:** These represents returns, corrections, or free items that distort revenue calculations.\n"
        "4. **Convert CustomerID to string** and format **InvoiceDate** as datetime.\n"
        "5. **Create TotalAmount** feature: `Quantity * UnitPrice`."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# 1. Drop rows with null CustomerID\n"
        "clean_df = df.dropna(subset=['CustomerID']).copy()\n"
        "\n"
        "# 2. Remove cancelled invoices (InvoiceNo starting with 'C')\n"
        "clean_df = clean_df[~clean_df['InvoiceNo'].astype(str).str.startswith('C')]\n"
        "\n"
        "# 3. Remove non-positive quantities and prices\n"
        "clean_df = clean_df[(clean_df['Quantity'] > 0) & (clean_df['UnitPrice'] > 0)]\n"
        "\n"
        "# 4. Format columns\n"
        "clean_df['CustomerID'] = clean_df['CustomerID'].astype(int).astype(str)\n"
        "clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])\n"
        "\n"
        "# 5. Calculate Total Amount spent per transaction item\n"
        "clean_df['TotalAmount'] = clean_df['Quantity'] * clean_df['UnitPrice']\n"
        "\n"
        "print(f\"Post-Cleaning shape: {clean_df.shape[0]:,} rows (filtered {(df.shape[0]-clean_df.shape[0]):,} rows)\")\n"
        "print(f\"Unique Customers: {clean_df['CustomerID'].nunique():,}\")\n"
        "print(f\"Unique Products: {clean_df['StockCode'].nunique():,}\")"
    ))
    
    # 6. Geographical Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Geographical Distribution Analysis\n"
        "Let's see which countries have the highest transaction volumes and total revenue."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Group transactions and revenue by country\n"
        "country_stats = clean_df.groupby('Country').agg(\n"
        "    TransactionCount=('InvoiceNo', 'count'),\n"
        "    TotalRevenue=('TotalAmount', 'sum')\n"
        ").sort_values(by='TotalRevenue', ascending=False).reset_index()\n"
        "\n"
        "# Add percentage columns\n"
        "country_stats['TransactionShare (%)'] = (country_stats['TransactionCount'] / country_stats['TransactionCount'].sum() * 100).round(2)\n"
        "country_stats['RevenueShare (%)'] = (country_stats['TotalRevenue'] / country_stats['TotalRevenue'].sum() * 100).round(2)\n"
        "\n"
        "print(\"Top 10 Countries by Revenue:\")\n"
        "display(country_stats.head(10))"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Plot transaction volumes and revenues side-by-side\n"
        "fig, axes = plt.subplots(1, 2, figsize=(20, 6))\n"
        "\n"
        "sns.barplot(data=country_stats.head(10), x='TransactionCount', y='Country', palette='viridis', ax=axes[0])\n"
        "axes[0].set_title('Top 10 Countries by Transaction Count')\n"
        "axes[0].set_xlabel('Number of Transactions')\n"
        "axes[0].set_ylabel('Country')\n"
        "\n"
        "sns.barplot(data=country_stats.head(10), x='TotalRevenue', y='Country', palette='magma', ax=axes[1])\n"
        "axes[1].set_title('Top 10 Countries by Total Revenue')\n"
        "axes[1].set_xlabel('Total Revenue ($)')\n"
        "axes[1].set_ylabel('')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 7. Top Selling Products
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Product Sales Insights\n"
        "Identify the top-selling products by total Quantity sold and total Revenue generated."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Clean descriptions (trim whitespace and make uppercase)\n"
        "clean_df['Description'] = clean_df['Description'].str.strip().str.upper()\n"
        "\n"
        "# Top products by quantity sold\n"
        "top_qty = clean_df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()\n"
        "\n"
        "# Top products by revenue\n"
        "top_revenue = clean_df.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10).reset_index()\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(22, 7))\n"
        "\n"
        "sns.barplot(data=top_qty, x='Quantity', y='Description', palette='Blues_r', ax=axes[0])\n"
        "axes[0].set_title('Top 10 Products by Quantity Sold')\n"
        "axes[0].set_xlabel('Units Sold')\n"
        "axes[0].set_ylabel('Product Description')\n"
        "\n"
        "sns.barplot(data=top_revenue, x='TotalAmount', y='Description', palette='Oranges_r', ax=axes[1])\n"
        "axes[1].set_title('Top 10 Products by Revenue Generated')\n"
        "axes[1].set_xlabel('Revenue ($)')\n"
        "axes[1].set_ylabel('')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 8. Temporal Patterns
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Temporal Sales & Transaction Patterns\n"
        "Let's extract and plot transaction trends across months, days of the week, and hours of the day "
        "to discover when customers are most active."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Extract monthly and weekly variables\n"
        "clean_df['Month'] = clean_df['InvoiceDate'].dt.to_period('M').astype(str)\n"
        "clean_df['DayOfWeek'] = clean_df['InvoiceDate'].dt.day_name()\n"
        "clean_df['Hour'] = clean_df['InvoiceDate'].dt.hour\n"
        "\n"
        "# Aggregate monthly revenue and invoice counts\n"
        "monthly_data = clean_df.groupby('Month').agg(\n"
        "    Revenue=('TotalAmount', 'sum'),\n"
        "    Invoices=('InvoiceNo', 'nunique')\n"
        ").reset_index()\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(15, 6))\n"
        "sns.lineplot(data=monthly_data, x='Month', y='Revenue', marker='o', color='royalblue', linewidth=2.5, ax=ax1)\n"
        "ax1.set_title('Monthly Revenue and Transaction Trends', fontsize=14)\n"
        "ax1.set_xlabel('Month', fontsize=12)\n"
        "ax1.set_ylabel('Revenue ($)', color='royalblue', fontsize=12)\n"
        "ax1.tick_params(axis='y', labelcolor='royalblue')\n"
        "plt.xticks(rotation=45)\n"
        "\n"
        "ax2 = ax1.twinx()\n"
        "sns.lineplot(data=monthly_data, x='Month', y='Invoices', marker='s', color='forestgreen', linewidth=2.5, ax=ax2)\n"
        "ax2.set_ylabel('Transaction Volume (Invoices)', color='forestgreen', fontsize=12)\n"
        "ax2.tick_params(axis='y', labelcolor='forestgreen')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Sales by Day of the Week\n"
        "# Note: Sunday has no transactions/data in this retail dataset\n"
        "day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']\n"
        "weekly_revenue = clean_df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order).reset_index()\n"
        "\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.barplot(data=weekly_revenue, x='DayOfWeek', y='TotalAmount', palette='crest')\n"
        "plt.title('Revenue by Day of Week')\n"
        "plt.xlabel('Day of the Week')\n"
        "plt.ylabel('Revenue ($)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Sales by Hour of Day\n"
        "hourly_revenue = clean_df.groupby('Hour')['TotalAmount'].sum().reset_index()\n"
        "\n"
        "plt.figure(figsize=(12, 5))\n"
        "sns.lineplot(data=hourly_revenue, x='Hour', y='TotalAmount', marker='o', color='darkorange', linewidth=2.5)\n"
        "plt.title('Revenue Distribution by Hour of the Day')\n"
        "plt.xlabel('Hour of Day (24-hour format)')\n"
        "plt.ylabel('Revenue ($)')\n"
        "plt.xticks(range(6, 21))\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 9. Distributions
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Distributonal Characteristics of Transactions\n"
        "We inspect the distributions of unit price, quantity, and total transaction values to spot outliers "
        "and skewness."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(18, 5))\n"
        "\n"
        "# UnitPrice Distribution\n"
        "sns.histplot(clean_df[clean_df['UnitPrice'] < 15]['UnitPrice'], bins=30, kde=True, color='teal', ax=axes[0])\n"
        "axes[0].set_title('UnitPrice Distribution (Filtered < $15)')\n"
        "axes[0].set_xlabel('Unit Price ($)')\n"
        "\n"
        "# Quantity Distribution\n"
        "sns.histplot(clean_df[clean_df['Quantity'] < 40]['Quantity'], bins=40, kde=True, color='darkmagenta', ax=axes[1])\n"
        "axes[1].set_title('Quantity Distribution per Line Item (Filtered < 40)')\n"
        "axes[1].set_xlabel('Quantity Purchased')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Invoice value distribution\n"
        "invoice_totals = clean_df.groupby('InvoiceNo')['TotalAmount'].sum()\n"
        "plt.figure(figsize=(10, 5))\n"
        "sns.histplot(invoice_totals[invoice_totals < 1200], bins=50, kde=True, color='tomato')\n"
        "plt.title('Distribution of Invoice Order Value (Clipped at $1200)')\n"
        "plt.xlabel('Order Value ($)')\n"
        "plt.ylabel('Frequency')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/EDA_Analysis.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("notebooks/EDA_Analysis.ipynb generated successfully!")

def create_clustering_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Header
    cells.append(nbf.v4.new_markdown_cell(
        "# 🎯 Customer Segmentation (RFM + KMeans) & Product Recommendations\n"
        "This notebook focuses on the modeling core of the **Shopper Spectrum** project. We will:\n"
        "1. Calculate RFM (Recency, Frequency, Monetary) metrics per customer.\n"
        "2. Preprocess features using the optimized **Log-Transform + StandardScaler** pipeline.\n"
        "3. Train and tune K-Means clustering ($K=4$), and dynamically map segments.\n"
        "4. Build an item-based collaborative filtering recommender using Cosine Similarity.\n"
        "5. Export models and similarity dictionaries to the `models/` directory."
    ))
    
    # 2. Imports
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from IPython.display import display\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import silhouette_score\n"
        "from sklearn.metrics.pairwise import cosine_similarity\n"
        "import joblib\n"
        "import os\n"
        "\n"
        "# Set styles\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (12, 6)\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')"
    ))
    
    # 3. Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Data Cleaning and Loading\n"
        "We load the transaction dataset from `../data/online_retail.csv` and filter out missing values, returns, and cancellations."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "csv_path = '../data/online_retail.csv'\n"
        "df = pd.read_csv(csv_path)\n"
        "\n"
        "# Clean\n"
        "clean_df = df.dropna(subset=['CustomerID']).copy()\n"
        "clean_df = clean_df[~clean_df['InvoiceNo'].astype(str).str.startswith('C')]\n"
        "clean_df = clean_df[(clean_df['Quantity'] > 0) & (clean_df['UnitPrice'] > 0)]\n"
        "clean_df['CustomerID'] = clean_df['CustomerID'].astype(int).astype(str)\n"
        "clean_df['InvoiceDate'] = pd.to_datetime(clean_df['InvoiceDate'])\n"
        "clean_df['TotalAmount'] = clean_df['Quantity'] * clean_df['UnitPrice']\n"
        "clean_df['Description'] = clean_df['Description'].str.strip().str.upper()\n"
        "\n"
        "print(f\"Cleaned dataframe contains {clean_df.shape[0]:,} rows for {clean_df['CustomerID'].nunique():,} customers.\")"
    ))
    
    # 4. RFM Feature Engineering
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. RFM Feature Engineering\n"
        "We engineer three critical customer behavior features:\n"
        "* **Recency ($R$):** Days since the customer's last transaction (relative to the dataset max date + 1 day).\n"
        "* **Frequency ($F$):** Total number of unique transactions (invoices) completed.\n"
        "* **Monetary ($M$):** Total dollar amount spent."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Establish reference date\n"
        "ref_date = clean_df['InvoiceDate'].max() + pd.Timedelta(days=1)\n"
        "print(\"Reference date:\", ref_date)\n"
        "\n"
        "# Calculate RFM metrics\n"
        "rfm = clean_df.groupby('CustomerID').agg(\n"
        "    Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),\n"
        "    Frequency=('InvoiceNo', 'nunique'),\n"
        "    Monetary=('TotalAmount', 'sum')\n"
        ").reset_index()\n"
        "\n"
        "print(\"RFM Table Head:\")\n"
        "display(rfm.head())\n"
        "print(\"\\nSummary Statistics:\")\n"
        "display(rfm.describe())"
    ))
    
    # 5. Data Transformation & Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Log-Transformation and Standardization\n"
        "Due to severe right skewness in frequency and monetary value, standardizing raw features directamente drags centroids "
        "towards high-value outliers and compresses the rest of the customer base. "
        "We apply a `log(x + 1)` transform to resolve the skewness, followed by `StandardScaler` to bring features to a standard scale."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Log Transform to normalize distribution\n"
        "rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])\n"
        "\n"
        "# Scale features\n"
        "scaler_log = StandardScaler()\n"
        "rfm_log_scaled = pd.DataFrame(\n"
        "    scaler_log.fit_transform(rfm_log), \n"
        "    columns=['Recency', 'Frequency', 'Monetary']\n"
        ")\n"
        "\n"
        "# Plot Log-transformed distributions\n"
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):\n"
        "    sns.histplot(rfm_log[col], kde=True, color='darkgreen', ax=axes[idx])\n"
        "    axes[idx].set_title(f\"Log-Transformed {col}\\nSkewness: {rfm_log[col].skew():.2f}\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 6. Hyperparameter Tuning
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Hyperparameter Tuning: Elbow Method & Silhouette Score\n"
        "We evaluate K-Means performance for $K \\in [2, 8]$ using Inertia (within-cluster sum of squares) and "
        "Silhouette Score (computed on a representative sample of 2,000 customers for efficiency)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "inertia = []\n"
        "silhouette_scores = []\n"
        "K_range = range(2, 9)\n"
        "\n"
        "for k in K_range:\n"
        "    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n"
        "    labels = km.fit_predict(rfm_log_scaled)\n"
        "    inertia.append(km.inertia_)\n"
        "    \n"
        "    # For speed, compute Silhouette score on a random sample of 2000 customers\n"
        "    sample_idx = rfm_log_scaled.sample(n=2000, random_state=42).index\n"
        "    silhouette_scores.append(silhouette_score(rfm_log_scaled.loc[sample_idx], labels[sample_idx]))\n"
        "\n"
        "fig, ax1 = plt.subplots(figsize=(14, 6))\n"
        "\n"
        "ax1.plot(list(K_range), inertia, 'b-x', linewidth=2, markersize=8)\n"
        "ax1.set_xlabel('Number of Clusters (K)')\n"
        "ax1.set_ylabel('Inertia (Sum of squared distances)', color='blue')\n"
        "ax1.tick_params(axis='y', labelcolor='blue')\n"
        "ax1.set_title('Elbow Curve & Silhouette Score Analysis for Log-Scaled KMeans')\n"
        "\n"
        "ax2 = ax1.twinx()\n"
        "ax2.plot(list(K_range), silhouette_scores, 'r-o', linewidth=2, markersize=8)\n"
        "ax2.set_ylabel('Silhouette Score (Higher is Better)', color='red')\n"
        "ax2.tick_params(axis='y', labelcolor='red')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 7. Final Model Fitting & Profiling
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Final KMeans Model ($K=4$) & Segment Characterization\n"
        "We fit our final model with $K=4$ and dynamically map the clusters to these standard segment names:\n"
        "* **High-Value:** Very recent purchases, high transaction frequency, and high monetary spend.\n"
        "* **Regular:** Frequent, steady spenders with moderate recency.\n"
        "* **Occasional:** Low transaction frequency, moderate monetary spend, and low days since last purchase.\n"
        "* **At-Risk:** Lapsed buyers who haven't shopped in a long time, with low frequency and low spend."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "final_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)\n"
        "rfm['Cluster'] = final_kmeans.fit_predict(rfm_log_scaled)\n"
        "\n"
        "# Profile each cluster\n"
        "profile = rfm.groupby('Cluster').agg(\n"
        "    Recency_Mean=('Recency', 'mean'),\n"
        "    Frequency_Mean=('Frequency', 'mean'),\n"
        "    Monetary_Mean=('Monetary', 'mean'),\n"
        "    Customer_Count=('Cluster', 'count')\n"
        ").reset_index()\n"
        "\n"
        "display(profile)"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Dynamic Mapper based on cluster centroids\n"
        "highest_monetary_cluster = profile.loc[profile['Monetary_Mean'].idxmax(), 'Cluster']\n"
        "highest_recency_cluster = profile.loc[profile['Recency_Mean'].idxmax(), 'Cluster']\n"
        "\n"
        "remaining_clusters = [c for c in [0,1,2,3] if c not in [highest_monetary_cluster, highest_recency_cluster]]\n"
        "\n"
        "if profile.loc[profile['Cluster'] == remaining_clusters[0], 'Monetary_Mean'].values[0] > \\\n"
        "   profile.loc[profile['Cluster'] == remaining_clusters[1], 'Monetary_Mean'].values[0]:\n"
        "    regular_cluster = remaining_clusters[0]\n"
        "    occasional_cluster = remaining_clusters[1]\n"
        "else:\n"
        "    regular_cluster = remaining_clusters[1]\n"
        "    occasional_cluster = remaining_clusters[0]\n"
        "\n"
        "mapping = {\n"
        "    highest_monetary_cluster: 'High-Value',\n"
        "    regular_cluster: 'Regular',\n"
        "    occasional_cluster: 'Occasional',\n"
        "    highest_recency_cluster: 'At-Risk'\n"
        "}\n"
        "\n"
        "rfm['Segment'] = rfm['Cluster'].map(mapping)\n"
        "print(\"Cluster Mapping:\", mapping)\n"
        "print(\"\\nSegment counts:\")\n"
        "print(rfm['Segment'].value_counts())"
    ))
    
    # 8. 3D Cluster Visualization
    cells.append(nbf.v4.new_markdown_cell(
        "### 3D Visualization of Customer Segments\n"
        "Let's visualize the customer segments in 3D space using Matplotlib."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig = plt.figure(figsize=(12, 9))\n"
        "ax = fig.add_subplot(111, projection='3d')\n"
        "\n"
        "colors = {'High-Value': 'gold', 'Regular': 'dodgerblue', 'Occasional': 'limegreen', 'At-Risk': 'crimson'}\n"
        "\n"
        "for segment, color in colors.items():\n"
        "    subset = rfm[rfm['Segment'] == segment]\n"
        "    ax.scatter(\n"
        "        np.log1p(subset['Recency']),\n"
        "        np.log1p(subset['Frequency']),\n"
        "        np.log1p(subset['Monetary']),\n"
        "        c=color, label=segment, s=40, alpha=0.6, edgecolors='w', linewidth=0.5\n"
        ")\n"
        "\n"
        "ax.set_xlabel('Log(Recency)')\n"
        "ax.set_ylabel('Log(Frequency)')\n"
        "ax.set_zlabel('Log(Monetary)')\n"
        "ax.set_title('3D Representation of Log-Transformed RFM Customer Segments')\n"
        "ax.legend(title='Customer Segment')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 9. Recommendation System
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Product Recommendation System (Item-Based Collaborative Filtering)\n"
        "To recommend products, we build a **Customer-Product Interaction Matrix**:\n"
        "1. Create a pivot table where rows are `CustomerID`, columns are product `Description`, and cell values are binary indicators (`1` if purchased, `0` otherwise).\n"
        "2. Compute the **Cosine Similarity** between the columns (products).\n"
        "3. Find the top 5 most similar products for a user-input product name."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Create customer-product pivot table with binary indicator\n"
        "pivot_df = clean_df.pivot_table(\n"
        "    index='CustomerID', columns='Description', values='Quantity', \n"
        "    aggfunc='sum', fill_value=0\n"
        ")\n"
        "pivot_binary = pivot_df.clip(upper=1)\n"
        "\n"
        "print(\"Interaction matrix shape:\", pivot_binary.shape)"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Compute Item similarity matrix\n"
        "item_similarity = cosine_similarity(pivot_binary.T)\n"
        "similarity_df = pd.DataFrame(item_similarity, index=pivot_binary.columns, columns=pivot_binary.columns)\n"
        "print(\"Item similarity matrix computed.\")"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Let's define the recommendation function\n"
        "def get_top_5_recommendations(product_name):\n"
        "    product_name = product_name.upper().strip()\n"
        "    if product_name not in similarity_df.index:\n"
        "        # Simple search for closest matches\n"
        "        matches = [idx for idx in similarity_df.index if product_name in idx]\n"
        "        if not matches:\n"
        "            return f\"Product '{product_name}' not found in catalog.\"\n"
        "        else:\n"
        "            print(f\"Product not found. Did you mean: {matches[:3]}?\")\n"
        "            product_name = matches[0]\n"
        "            print(f\"Recommending for closest match: '{product_name}'\\n\")\n"
        "            \n"
        "    # Get top 5 similarities (excluding itself)\n"
        "    similar_items = similarity_df[product_name].sort_values(ascending=False).iloc[1:6]\n"
        "    return pd.DataFrame({'Product': similar_items.index, 'Similarity Score': similar_items.values})"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Test recommendation for a popular item\n"
        "get_top_5_recommendations('WHITE HANGING HEART T-LIGHT HOLDER')"
    ))
    
    # 10. Model Exports & Optimizations
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Model Export & Similarity Dictionary Optimization\n"
        "We save standardizer, KMeans model, cluster mapping, and optimized top-N recommendations dictionary to the `../models/` directory."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Create the models directory if it doesn't exist\n"
        "os.makedirs('../models', exist_ok=True)\n"
        "\n"
        "# 1. Save standardizer and clustering model\n"
        "joblib.dump(scaler_log, '../models/scaler.pkl')\n"
        "joblib.dump(final_kmeans, '../models/kmeans_model.pkl')\n"
        "\n"
        "# 2. Save cluster to label mapping\n"
        "joblib.dump(mapping, '../models/cluster_mapping.pkl')\n"
        "\n"
        "# 3. Compute and save top 10 similarity dictionary\n"
        "similarity_dict = {}\n"
        "for col in similarity_df.columns:\n"
        "    top_10 = similarity_df[col].sort_values(ascending=False).iloc[1:11]\n"
        "    similarity_dict[col] = list(zip(top_10.index, top_10.values.round(4)))\n"
        "\n"
        "joblib.dump(similarity_dict, '../models/product_similarity.pkl')\n"
        "print(\"All modeling artifacts successfully serialized to the '../models/' directory.\")"
    ))
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/Customer_Segmentation_Clustering.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("notebooks/Customer_Segmentation_Clustering.ipynb generated successfully!")

if __name__ == '__main__':
    create_eda_notebook()
    create_clustering_notebook()
