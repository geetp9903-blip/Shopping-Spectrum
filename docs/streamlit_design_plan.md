# Streamlit Application Design Plan 📱

## Shopper Spectrum: Interactive Analytics, Segmentation, and Recommendations

This document outlines the UI/UX architecture and frontend design specification for the Streamlit application. The goal is to build a premium, highly responsive, and modern web interface that breaks away from generic Streamlit layouts by injecting custom styling, responsive cards, and interactive visualizations while maintaining the core business functionality.

---

## 1. Visual Identity & Theme Definition (Cyberpunk Analytics)

We will adopt a **High-Tech Cyberpunk** aesthetic, tailored for high-speed data interpretation and futuristic e-commerce monitoring. The brand personality is aggressive, precise, and cutting-edge, designed to evoke a "command center" emotional response, but balanced to ensure the glow effects are **not overpowering or distracting**.

### A. Color Palette

The palette is built on a "Void Black" base to maximize the luminance of neon accents subtly.

* **Background:** Void Black (`#0C0F0F`)
* **Card/Surface Background:** Dark Charcoal (`#161B1B`) with semi-transparent backdrop blur (`rgba(22, 27, 27, 0.7)`).
* **Primary Accent:** Neon Purple (`#BC13FE`) - Used for primary actions, branding, and high-level navigational states.
* **Secondary Accent:** Cyan (`#00FFFF`) - Reserved for data visualization, active selection indicators, and technical details.
* **Tertiary/Status:** Lime Green (`#32CD32`) - Specifically for positive growth metrics, success states, and "system online" indicators.
* **Glow/Borders:** To keep the glow from being "too much", cards will use a subtle 1px border with a 30% opacity glow effect, rather than intense heavy shadows.

### B. Custom Typography

We will use **Montserrat** (with a fallback to **Avenir Next**) for exceptional legibility and a **Monospace** font for raw data points to emphasize the technical nature of the tool.

```css
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: "Avenir Next", "Montserrat", -apple-system, sans-serif;
}

/* Monospace for metrics and data */
.metric-value, .data-point {
    font-family: "JetBrains Mono", monospace;
    letter-spacing: 0.05em;
}
```

---

## 2. Layout Structure & Navigation

The layout follows a rigid grid philosophy. Content should feel docked into a system rather than floating freely. The application will feature a sidebar navigation panel with high-quality tech-inspired icons representing each module:

```
Streamlit App
├── Sidebar (Navigation & Model Info, Neon Purple accents)
└── Main Panel (Dynamic Page Display)
    ├── Tab 1: 📈 Business Dashboard (EDA)
    ├── Tab 2: 🎯 Customer Segmentation (RFM)
    └── Tab 3: 🛒 Product Recommender (Collaborative Filtering)
```

---

## 3. Screen-by-Screen UI/UX Design

### Page 1: 📈 Business Dashboard (EDA Insights)

This page acts as the executive command center showing transaction statistics and retail trends.

* **Top KPI Row:** Displays 4 styled, dark glassmorphic metric cards containing:
  - **Total Revenue:** `$8.3M` (highlighted with Lime Green growth indicators)
  - **Total Invoices:** `18.5k`
  - **Active Customers:** `4.3k`
  - **Avg. Order Value:** `$362`
* **Plotly Interactive Charts (with Neon glows):**
  - **Geographical Share:** Horizontal bar chart using Cyan (`#00FFFF`) for the bars.
  - **Sales Over Time:** A line chart showing monthly sales trends using a Neon Purple (`#BC13FE`) line with a soft glow effect.
  - **Peak Purchase Windows:** A heatmap with neon gradient scaling.
  - **Top Product Catalog:** A slider control allowing the user to select how many top products to visualize.

### Page 2: 🎯 Customer Segmentation (RFM Predictor)

This page allows marketing teams to input customer metrics and instantly retrieve their segment profile.

* **Split Layout:**
  - **Left Column (Inputs):** Styled sliders and numerical text inputs with Cyan active borders.
    - **Recency:** Days since last purchase
    - **Frequency:** Total number of invoices
    - **Monetary:** Total amount spent
  - **Right Column (Predictions & Metrics):**
    - A dynamic "Predict" trigger button (Cyan fill, black text).
    - **Custom Prediction Card:** Once calculated, the card borders glow with the color of the predicted segment (Gold, Cyan, Green, or Red).
    - **Segment Centroid Comparison:** A bar chart comparing the input customer's values against the cluster's average centroid.

### Page 3: 🛒 Product Recommender

This module allows catalog managers to look up any inventory product and see recommendations.

* **Fuzzy Product Finder:**
  - Text input with a glowing bottom border (Cyan when focused) and fuzzy auto-suggestions.
* **Top 5 Recommendation Cards:**
  - Recommendations will be laid out in a grid of 5 styled product cards with a 0.25rem (4px) soft-sharp radius.
  - Each card contains:
    - Product title in Inter font.
    - Similarity score progress bar (e.g. `94% match` shown using a progress bar styled with Cyan or Lime Green).

---

## 4. Technical Stack for Streamlit UI

To implement these features cleanly, we will use the following Python imports and configurations:

1. **`streamlit`**: Base layout, sidebar controls, sliders.
2. **`plotly.express` & `plotly.graph_objects`**: Plotly supports dark themes out of the box (`plotly_dark` template) and we will customize line properties to add a soft `glow` effect (e.g., using `line=dict(color='#BC13FE', width=3)` with overlapping transparent thick lines if necessary).
3. **`pandas` & `numpy`**: Running underlying data lookups.
4. **`joblib`**: Quick loading of `scaler.pkl`, `kmeans_model.pkl`, `cluster_mapping.pkl`, and `product_similarity.pkl`.
5. **Custom CSS Injector:**
   ```python
   def local_css(file_name):
       with open(file_name) as f:
           st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
   ```

---

## 5. UI/UX Polishing Checklist

* [ ] **Balanced Glow:** Ensure `box-shadow` values for cards and buttons use low opacity (e.g., `rgba(188, 19, 254, 0.2)`) to keep the cyberpunk theme professional and not overly harsh.
* [ ] **Outlier Warnings:** If the user inputs a Monetary value of `$50,000+` in customer segmentation, display an info tooltip.
* [ ] **Interactive Autocomplete:** Ensure that typing immediately lists matching descriptions.
* [ ] **CSS Styling Sheet:** Keep all styling rules separated in a `assets/style.css` file.
