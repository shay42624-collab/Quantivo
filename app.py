import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Page config and title
st.set_page_config(page_title="Zynova Customer Segmentation", layout="wide")
st.title("📊 Zynova – AI-Powered Customer Segmentation")

# Upload CSV
uploaded_file = st.file_uploader("Upload your customer_data.csv", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Encode categorical columns
    for col in ['gender', 'region']:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    # Handle missing values
    data.fillna(data.median(numeric_only=True), inplace=True)

    # Select and scale numerical features
    num_cols = ['age', 'annual_income', 'spending_score']
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['segment'] = kmeans.fit_predict(data[num_cols])

    # PCA for visualization
    st.subheader("📍 Segment Visualization")
    pca = PCA(n_components=2)
    components = pca.fit_transform(data[num_cols])
    fig, ax = plt.subplots()
    scatter = ax.scatter(components[:, 0], components[:, 1], c=data['segment'], cmap='viridis')
    ax.set_title("Customer Segments")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

    # Segment insights
    st.subheader("📈 Segment Insights")
    for segment in sorted(data['segment'].unique()):
        st.markdown(f"**Segment {segment} stats:**")
        st.dataframe(data[data['segment'] == segment][num_cols].mean().round(2))


