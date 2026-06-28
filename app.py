import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import textwrap
from src.data_loader import load_data, load_models

# Set Page Config
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to inject custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file {file_name} not found.")

# Inject the Cyberpunk styling
local_css("assets/style.css")

# Load real dataset
df = load_data()

if df.empty:
    st.error("Failed to load dataset. Please check data/online_retail.csv")
    st.stop()

# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
st.sidebar.title("Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📈 Business Dashboard", "🎯 Customer Segmentation", "🛒 Product Recommender"])

# ------------------------------------------------------------------------------
# Main Dashboard
# ------------------------------------------------------------------------------
if page == "📈 Business Dashboard":
    st.markdown("<h1>Business Insights</h1>", unsafe_allow_html=True)
    
    # Calculate Real Metrics
    total_revenue = df['TotalPrice'].sum()
    total_invoices = df['InvoiceNo'].nunique()
    total_customers = df['CustomerID'].nunique()
    aov = total_revenue / total_invoices if total_invoices > 0 else 0
    
    # Formatted values
    rev_str = f"${total_revenue/1e6:.2f}M"
    inv_str = f"{total_invoices/1e3:.1f}k"
    cust_str = f"{total_customers/1e3:.1f}k"
    aov_str = f"${aov:.0f}"
    
    # KPIs styled perfectly inside custom HTML (Fixing the empty rectangles)
    kpi_html = f"""
    <div class="kpi-container">
        <div class="cyber-card kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{rev_str}</div>
        </div>
        <div class="kpi-card cyber-card">
            <div class="kpi-label">Total Invoices</div>
            <div class="kpi-value">{inv_str}</div>
        </div>
        <div class="kpi-card cyber-card">
            <div class="kpi-label">Active Customers</div>
            <div class="kpi-value">{cust_str}</div>
        </div>
        <div class="kpi-card cyber-card">
            <div class="kpi-label">Avg. Order Value</div>
            <div class="kpi-value">{aov_str}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("<h3>Monthly Revenue Trends</h3>", unsafe_allow_html=True)
        # Aggregate real data by month and format values for data labels
        df['Month'] = df['InvoiceDate'].dt.to_period('M').dt.start_time
        df_trends = df.groupby('Month')['TotalPrice'].sum().reset_index()
        df_trends['FormattedPrice'] = df_trends['TotalPrice'].apply(lambda x: f"${x/1e3:.0f}k" if x < 1e6 else f"${x/1e6:.2f}M")
        
        # Line Chart with data labels
        fig_line = px.line(df_trends, x='Month', y='TotalPrice', text='FormattedPrice', markers=True, template="plotly_dark")

        # Apply Cyberpunk Purple
        fig_line.update_traces(
            line=dict(color="#BC13FE", width=3), 
            marker=dict(size=8, color="#00FFFF"),
            textposition="top center"
        )
        # Using a solid background for the chart to act as a card itself
        fig_line.update_layout(
            paper_bgcolor="#161B1B", 
            plot_bgcolor="#161B1B",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, df_trends['TotalPrice'].max() * 1.15]) # Extra vertical padding for data labels
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_chart2:
        st.markdown("<h3>Geographical Share</h3>", unsafe_allow_html=True)
        # Aggregate real data by country (Top 10)
        df_geo = df.groupby('Country')['TotalPrice'].sum().reset_index()
        df_geo = df_geo.sort_values('TotalPrice', ascending=True).tail(10)
        df_geo['FormattedPrice'] = df_geo['TotalPrice'].apply(lambda x: f"${x/1e3:.0f}k" if x < 1e6 else f"${x/1e6:.2f}M")
        
        # Bar Chart
        fig_bar = px.bar(df_geo, x='TotalPrice', y='Country', text='FormattedPrice', orientation='h', template="plotly_dark")
        # Apply Cyberpunk Cyan
        fig_bar.update_traces(
            marker_color="#00FFFF",
            textposition="outside"
        )
        fig_bar.update_layout(
            paper_bgcolor="#161B1B", 
            plot_bgcolor="#161B1B",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, df_geo['TotalPrice'].max() * 1.2]) # 20% range padding for text labels on the right
        )
        st.plotly_chart(fig_bar, use_container_width=True)

elif page == "🎯 Customer Segmentation":
    st.markdown("<h1>Customer Segmentation Predictor</h1>", unsafe_allow_html=True)
    st.markdown("Enter customer metrics below to predict their loyalty segment.", unsafe_allow_html=True)
    
    scaler, kmeans_model, cluster_mapping, _ = load_models()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (Days since last purchase)", min_value=1, max_value=1000, value=30)
    with col2:
        frequency = st.number_input("Frequency (Number of purchases)", min_value=1, max_value=1000, value=5)
    with col3:
        monetary = st.number_input("Monetary Value ($)", min_value=1.0, max_value=100000.0, value=500.0, step=10.0)
        
    if st.button("Predict Segment", use_container_width=True):
        if scaler and kmeans_model and cluster_mapping:
            # Apply log1p transform (as trained)
            r_log = np.log1p(recency)
            f_log = np.log1p(frequency)
            m_log = np.log1p(monetary)
            
            # Format DataFrame to prevent sklearn feature-name warnings
            input_df = pd.DataFrame([[r_log, f_log, m_log]], columns=['Recency', 'Frequency', 'Monetary'])
            
            scaled_data = scaler.transform(input_df)
            cluster = kmeans_model.predict(scaled_data)[0]
            segment_name = cluster_mapping.get(cluster, "Unknown")
            
            # Define Personas
            personas = {
                "High-Value": ("Highly engaged VIPs. High spend, high frequency.", "Enroll in VIP loyalty program and offer early-access products."),
                "Regular": ("Consistent shoppers with moderate spend.", "Upsell with product bundles to increase AOV."),
                "Occasional": ("Low frequency, moderate spend.", "Send re-engagement discounts and seasonal promotions."),
                "At-Risk": ("Haven't purchased in a long time.", "Send aggressive win-back email sequences.")
            }
            desc, action = personas.get(segment_name, ("Unknown", "None"))
            
            color_class = "cyber-card-cyan"
            radar_color = "#00FFFF"
            fill_c = "rgba(0, 255, 255, 0.4)"
            
            if segment_name == "High-Value":
                color_class = "cyber-card-gold" 
                radar_color = "#FFD700"
                fill_c = "rgba(255, 215, 0, 0.4)"
            elif segment_name == "At-Risk":
                color_class = "cyber-card-red" 
                radar_color = "#FF4C4C"
                fill_c = "rgba(255, 76, 76, 0.4)"
                
            # Radar Chart logic
            categories = ['Recency', 'Frequency', 'Monetary']
            user_values = scaled_data[0].tolist()
            centroid_values = kmeans_model.cluster_centers_[cluster].tolist()
            
            # Close the radar loops
            user_values += [user_values[0]]
            centroid_values += [centroid_values[0]]
            categories_loop = categories + [categories[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=centroid_values,
                theta=categories_loop,
                fill='toself',
                name='Segment Average',
                line=dict(color='rgba(255, 255, 255, 0.3)'),
                fillcolor='rgba(255, 255, 255, 0.1)'
            ))
            fig.add_trace(go.Scatterpolar(
                r=user_values,
                theta=categories_loop,
                fill='toself',
                name='Customer Input',
                line=dict(color=radar_color),
                fillcolor=fill_c
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[-3, 3], gridcolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                paper_bgcolor="#161B1B",
                plot_bgcolor="#161B1B",
                template="plotly_dark",
                showlegend=True,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            # Render layout
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                html_str = f"""
                <div class="cyber-card {color_class} pulse-anim" style="height: 80%; display: flex; flex-direction: column; justify-content: center;">
                    <h3 style="margin-bottom: 0; opacity: 0.8;">Predicted Segment</h3>
                    <h1 style="color: {radar_color} !important; font-size: 2.5rem; margin-top: 0.5rem; margin-bottom: 1rem; text-shadow: 0 0 12px {fill_c};">{segment_name}</h1>
                    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><b>Profile:</b> {desc}</p>
                    <p style="font-size: 1.1rem; color: var(--accent-green);"><b>Action:</b> {action}</p>
                </div>
                """
                st.markdown(textwrap.dedent(html_str), unsafe_allow_html=True)
                
            with col_res2:
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Models failed to load.")
            
    # Always display Segment Benchmarks below the predictor
    benchmarks_html = """
    <h2 style="margin-top: 2.5rem; margin-bottom: 0.5rem;">Customer Segment Benchmarks</h2>
    <p style="margin-bottom: 1.5rem; color: var(--text-muted);">Reference profiles and typical metric ranges (IQR: 25th to 75th percentile) for all four loyalty segments.</p>
    
    <div class="benchmark-container">
        <!-- 1. High-Value -->
        <div class="benchmark-card high-value">
            <div class="benchmark-info">
                <div class="benchmark-title-row">
                    <span class="benchmark-name">High-Value</span>
                    <span class="benchmark-badge">732 customers (16.9%)</span>
                </div>
                <p class="benchmark-desc">Highly engaged VIPs. Exceptional spend, high frequency, and recently active.</p>
            </div>
            <div class="benchmark-metrics">
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Recency (Median)</span>
                    <span class="benchmark-metric-val">7.0 days</span>
                    <span class="benchmark-metric-range">Typical: 2.0 - 15.0d</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Frequency (Median)</span>
                    <span class="benchmark-metric-val">10.0 orders</span>
                    <span class="benchmark-metric-range">Typical: 7.0 - 15.0</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Monetary (Median)</span>
                    <span class="benchmark-metric-val">$3,699.7</span>
                    <span class="benchmark-metric-range">Typical: $2,314.3 - $6.3k</span>
                </div>
            </div>
            <div class="benchmark-action-box">
                <span class="benchmark-action-label">Marketing Strategy</span>
                <span class="benchmark-action-pill">VIP Loyalty & Early Access</span>
            </div>
        </div>
        
        <!-- 2. Regular -->
        <div class="benchmark-card regular">
            <div class="benchmark-info">
                <div class="benchmark-title-row">
                    <span class="benchmark-name">Regular</span>
                    <span class="benchmark-badge">1,152 customers (26.6%)</span>
                </div>
                <p class="benchmark-desc">Consistent shoppers with moderate spend. Highly loyal core customer base.</p>
            </div>
            <div class="benchmark-metrics">
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Recency (Median)</span>
                    <span class="benchmark-metric-val">56.0 days</span>
                    <span class="benchmark-metric-range">Typical: 30.8 - 89.0d</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Frequency (Median)</span>
                    <span class="benchmark-metric-val">4.0 orders</span>
                    <span class="benchmark-metric-range">Typical: 3.0 - 5.0</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Monetary (Median)</span>
                    <span class="benchmark-metric-val">$1,352.4</span>
                    <span class="benchmark-metric-range">Typical: $924.0 - $2.0k</span>
                </div>
            </div>
            <div class="benchmark-action-box">
                <span class="benchmark-action-label">Marketing Strategy</span>
                <span class="benchmark-action-pill">Upsell & Product Bundles</span>
            </div>
        </div>
        
        <!-- 3. Occasional -->
        <div class="benchmark-card occasional">
            <div class="benchmark-info">
                <div class="benchmark-title-row">
                    <span class="benchmark-name">Occasional</span>
                    <span class="benchmark-badge">857 customers (19.8%)</span>
                </div>
                <p class="benchmark-desc">Active recently but low purchase frequency. Moderate overall spending.</p>
            </div>
            <div class="benchmark-metrics">
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Recency (Median)</span>
                    <span class="benchmark-metric-val">16.0 days</span>
                    <span class="benchmark-metric-range">Typical: 8.0 - 25.0d</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Frequency (Median)</span>
                    <span class="benchmark-metric-val">2.0 orders</span>
                    <span class="benchmark-metric-range">Typical: 1.0 - 3.0</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Monetary (Median)</span>
                    <span class="benchmark-metric-val">$471.7</span>
                    <span class="benchmark-metric-range">Typical: $281.9 - $718.1</span>
                </div>
            </div>
            <div class="benchmark-action-box">
                <span class="benchmark-action-label">Marketing Strategy</span>
                <span class="benchmark-action-pill">Discounts & Seasonals</span>
            </div>
        </div>
        
        <!-- 4. At-Risk -->
        <div class="benchmark-card at-risk">
            <div class="benchmark-info">
                <div class="benchmark-title-row">
                    <span class="benchmark-name">At-Risk</span>
                    <span class="benchmark-badge">1,597 customers (36.8%)</span>
                </div>
                <p class="benchmark-desc">Haven't purchased in a long time. High chance of permanent churn.</p>
            </div>
            <div class="benchmark-metrics">
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Recency (Median)</span>
                    <span class="benchmark-metric-val">177.0 days</span>
                    <span class="benchmark-metric-range">Typical: 84.0 - 265.0d</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Frequency (Median)</span>
                    <span class="benchmark-metric-val">1.0 orders</span>
                    <span class="benchmark-metric-range">Typical: 1.0 - 2.0</span>
                </div>
                <div class="benchmark-metric-item">
                    <span class="benchmark-metric-label">Monetary (Median)</span>
                    <span class="benchmark-metric-val">$298.1</span>
                    <span class="benchmark-metric-range">Typical: $165.0 - $437.7</span>
                </div>
            </div>
            <div class="benchmark-action-box">
                <span class="benchmark-action-label">Marketing Strategy</span>
                <span class="benchmark-action-pill">Win-Back email sequences</span>
            </div>
        </div>
    </div>
    """
    # Strip all leading spaces so markdown parser doesn't treat nested divs as code blocks
    cleaned_html = "\n".join([line.strip() for line in benchmarks_html.split("\n")])
    st.markdown(cleaned_html, unsafe_allow_html=True)



elif page == "🛒 Product Recommender":
    st.markdown("<h1>Product Recommender Engine</h1>", unsafe_allow_html=True)
    
    _, _, _, product_similarity = load_models()
    
    if product_similarity:
        products = list(product_similarity.keys())
        selected_product = st.selectbox("Search for a product to find similar items:", products, index=0)
        
        if st.button("Get Recommendations", use_container_width=True):
            recommendations = product_similarity.get(selected_product, [])
            
            if recommendations:
                st.markdown("<h3>Top 5 Recommended Products:</h3>", unsafe_allow_html=True)
                
                html_cards = ""
                for rec_prod, score in recommendations[:5]:
                    pct_val = score * 100
                    card = f"""
                    <div class="recommender-row">
                        <div class="recommender-product-name">{rec_prod}</div>
                        <div class="recommender-bar-wrapper">
                            <div class="recommender-bar" style="width: {pct_val:.1f}%;"></div>
                        </div>
                        <span class="recommender-score">{pct_val:.1f}%</span>
                    </div>
                    """
                    html_cards += textwrap.dedent(card)
                
                container_html = f'<div class="recommender-container">\n{html_cards}\n</div>'
                st.markdown(container_html, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found for this product.")
    else:
        st.error("Similarity matrix failed to load.")
