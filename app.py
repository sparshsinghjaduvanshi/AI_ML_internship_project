import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecasting & Demand Intelligence Dashboard")

st.caption(
    "Superstore Sales Analysis using SARIMA, Prophet, XGBoost, Isolation Forest and K-Means Clustering"
)

st.markdown(
    """
    This dashboard presents sales analysis, forecasting,
    anomaly detection, and product demand segmentation
    using Machine Learning.
    """
)

st.divider()

df = pd.read_csv("train.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="mixed",
    dayfirst=True
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="mixed",
    dayfirst=True
)

df["Shipping Days"] = (
    df["Ship Date"] -
    df["Order Date"]
).dt.days

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Sales Overview",
        "📈 Forecast Explorer",
        "🚨 Anomaly Report",
        "📦 Product Segments"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dataset:** Superstore Sales
    
    **Forecast Models**
    - SARIMA
    - Prophet
    - XGBoost
    
    **Anomaly Detection**
    - Isolation Forest
    - Z-Score
    
    **Clustering**
    - K-Means + PCA
    """
)


total_sales = df["Sales"].sum()

total_orders = len(df)

avg_shipping = df["Shipping Days"].mean()

categories = df["Category"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

with col2:
    st.metric(
        "📦 Orders",
        f"{total_orders:,}"
    )

with col3:
    st.metric(
        "🚚 Avg Shipping",
        f"{avg_shipping:.1f} Days"
    )

with col4:
    st.metric(
        "🛒 Categories",
        categories
    )
    
if page == "🏠 Sales Overview":

    st.header("Sales Overview")

    with st.expander("📄 View Dataset"):

        st.dataframe(
        df.head(20),
        width="stretch"
    )
    # Monthly Sales
    monthly_sales = (
    df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
      .sum()
      .reset_index()
    )

    fig = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    st.plotly_chart(fig, width="stretch")    
    category_sales = (
        df.groupby("Category")["Sales"]
          .sum()
          .reset_index()
    )

    fig2 = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category"
    )

    
    region_sales = (
        df.groupby("Region")["Sales"]
          .sum()
          .reset_index()
    )
    
    fig3 = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        title="Regional Sales Distribution"
    )
    
    # st.plotly_chart(fig2, width="stretch")
    # st.plotly_chart(fig3, width="stretch")
    
    left, right = st.columns(2)

    with left:
        st.plotly_chart(fig2, width="stretch")

    with right:
        st.plotly_chart(fig3, width="stretch")
    
    top_products = (
        df.groupby("Product Name")["Sales"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
          .reset_index()
    )

    fig4 = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        title="Top 10 Products"
    )
    

    st.plotly_chart(fig4, width="stretch")
    
    
elif page == "📈 Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write("### Model Performance")

    metrics = pd.DataFrame({
        "Model": ["SARIMA", "Prophet", "XGBoost"],
        "MAE": [18031.40, 20250.79, 13915.32],
        "RMSE": [19009.18, 22318.41, 18893.85],
        "MAPE": [0.19, 0.22, 0.13]
    })

    st.dataframe(metrics, width="stretch")


    st.image(
        "charts/xgboost_forecast.png",
        caption="XGBoost Forecast",
        width="stretch" 
    )
    
    best_model = metrics.loc[metrics["RMSE"].idxmin()]

    st.success(
        f"🏆 Best Model: {best_model['Model']} (RMSE = {best_model['RMSE']:.2f})"
    )

    st.image(
        "charts/category_region_forecast.png",
        caption="Category & Region Forecast",
        width="stretch"     
    )

elif page == "🚨 Anomaly Report":

    st.header("🚨 Anomaly Detection Report")

    st.markdown("""
    This section highlights unusual sales patterns detected using two different techniques:

    - **Isolation Forest** (Machine Learning)
    - **Z-Score Method** (Statistical)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "charts/isolation_forest.png",
            caption="Isolation Forest Anomaly Detection",
            width="stretch"
        )

    with col2:
        st.image(
            "charts/zscore_anomalies.png",
            caption="Z-Score Based Anomaly Detection",
            width="stretch"
        )

    st.subheader("Business Interpretation")

    st.markdown("""
    - Isolation Forest detects both unusually **high** and **low** sales.
    - Z-Score mainly detects **extremely high** sales weeks.
    - Several anomalies appear during **November–December 2018**, indicating seasonal demand spikes.
    - These insights help improve inventory planning and promotional strategies.
    """)
    
    
elif page == "📦 Product Segments":

    st.header("📦 Product Demand Segmentation")

    st.markdown("""
    Products were grouped using **K-Means Clustering** based on:

    - Total Sales
    - Growth Rate
    - Sales Volatility
    - Average Order Value
    """)


    cluster_table = pd.DataFrame({
        "Demand Segment": [
            "High Growth Premium Products",
            "High Volume Stable Demand",
            "Low Volume Regular Demand",
            "Declining Premium Products"
        ],
        "Recommended Strategy": [
            "Increase inventory and monitor demand closely.",
            "Maintain regular stock levels.",
            "Keep moderate inventory to reduce holding costs.",
            "Reduce inventory and review pricing strategy."
        ]
    })

    st.image(
        "charts/product_clusters.png",
        caption="K-Means Product Segmentation",
        width="stretch"
    )
    st.subheader("Recommended Stocking Strategy")

    st.dataframe(cluster_table, width="stretch")

st.divider()

st.caption(
    "Developed by Sparsh Singh | End-to-End Sales Forecasting & Demand Intelligence System"
)