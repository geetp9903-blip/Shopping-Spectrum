# Full Scale Implementation Plan: Shopper Spectrum 🚀

## Goal Description
Implement the complete Shopper Spectrum Streamlit Application incorporating the established Cyberpunk UI/UX. This involves integrating the trained machine learning models, dynamically processing the real `online_retail.csv` dataset, and building out the functional interactive modules for Business Insights, Customer Segmentation, and Product Recommendations.

## User Review Required

> [!WARNING]
> **Performance Consideration:** The `online_retail.csv` dataset is roughly 48MB. We will need to use Streamlit's `@st.cache_data` effectively to prevent the app from reloading this dataset on every user interaction. 

## Proposed Changes

### 1. Data & Model Architecture (`src/data_loader.py`)
*   **[NEW]** Create a `src` directory and a `data_loader.py` utility.
*   **Dataset Loading:** Implement a function to load `data/online_retail.csv`, clean missing CustomerIDs, and calculate total price (`Quantity * UnitPrice`). This function will be decorated with `@st.cache_data`.
*   **Model Loading:** Implement functions decorated with `@st.cache_resource` to load the `.pkl` files from the `models/` directory using `joblib` (`scaler.pkl`, `kmeans_model.pkl`, `cluster_mapping.pkl`, `product_similarity.pkl`).

### 2. Module 1: Business Dashboard (`app.py`)
*   **[MODIFY]** Replace the dummy data in the `app.py` dashboard with live aggregations from the retail dataset.
*   **Metrics:** Calculate real Total Revenue, Invoices, Customers, and AOV, along with their Month-over-Month percentage changes.
*   **Visualizations:** 
    *   Aggregate revenue by month to feed the Monthly Trends line chart.
    *   Aggregate revenue by country to feed the Geographical Share bar chart.
    *   Ensure Plotly charts continue to use the Cyberpunk styling (Neon Purple, Cyan, transparent backgrounds).

### 3. Module 2: Customer Segmentation (`app.py`)
*   **[MODIFY]** Build the UI for Tab 2.
*   **Inputs:** Add Streamlit sliders and number inputs for Recency, Frequency, and Monetary value.
*   **Inference Pipeline:**
    *   Apply the loaded `scaler.pkl` to the user inputs.
    *   Pass the scaled data to `kmeans_model.pkl` to get the cluster ID.
    *   Map the ID using `cluster_mapping.pkl`.
*   **Dynamic UI Output:** Render a custom HTML card that changes its glow color based on the predicted segment (e.g., Gold for VIP, Red for At-Risk) using Python f-strings injected into the HTML template.

### 4. Module 3: Product Recommender (`app.py`)
*   **[MODIFY]** Build the UI for Tab 3.
*   **Search Interface:** Implement a fuzzy search text input where users can type a product name.
*   **Recommendation Engine:** Look up the selected product in the pre-computed `product_similarity.pkl` matrix.
*   **UI Output:** Render the Top 5 recommended products as a flexbox grid of smaller Cyber Cards, utilizing CSS progress bars to visualize the similarity score (0-100%).

### 5. UI/UX Final Polish (`assets/style.css`)
*   **[MODIFY]** Update `style.css` to include custom styling for Streamlit native inputs (sliders, text inputs) so they match the Void Black and Cyan Cyberpunk aesthetic.
*   **Animations:** Add subtle CSS pulse animations for the prediction output cards.

## Verification Plan

### Automated / Logic Tests
*   Run the Python scripts to verify `joblib` loads the models without versioning conflicts.
*   Check that the DataFrame aggregations produce the expected output shapes.

### Manual Verification
1.  **Data Load Time:** Verify the app starts up and caches the 48MB dataset in under 5 seconds.
2.  **Segmentation Inference:** Test edge-case inputs (e.g., 0 frequency, $1M monetary) to ensure the model doesn't crash and outputs a valid segment.
3.  **Recommender Test:** Search for a common item like "HEART T-LIGHT HOLDER" and ensure recommendations populate correctly.
4.  **Responsive UI:** Resize the browser window to mobile width to confirm the sidebar hides and the KPI flexbox wraps gracefully.
