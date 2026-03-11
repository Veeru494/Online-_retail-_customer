%%writefile app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout='wide')

st.title('RFM Customer Segmentation Dashboard')

# Load the RFM data
@st.cache_data
def load_data():
    rfm_df = pd.read_csv('rfm_data.csv', index_col='Customer ID')
    return rfm_df

rfm = load_data()

st.subheader('1. RFM Data Overview')
st.write(rfm.head())

# Prepare data for clustering and visualization (re-run steps for reproducibility within app)
scaler = StandardScaler()
# Select only RFM features for scaling
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]
rfm_scaled = scaler.fit_transform(rfm_features)

# Apply KMeans (using the same number of clusters as in analysis)
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
rfm['Cluster_KMeans'] = kmeans.fit_predict(rfm_scaled)

# Reduce dimensions with PCA for visualization
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
rfm['PC1'] = rfm_pca[:, 0]
rfm['PC2'] = rfm_pca[:, 1]

st.subheader('2. Customer Clusters (KMeans via PCA)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_KMeans', data=rfm, palette='viridis', legend='full', ax=ax)
ax.set_title('Customer Segmentation (KMeans via PCA)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
st.pyplot(fig)

st.subheader('3. Cluster Characteristics')
cluster_means = rfm.groupby('Cluster_KMeans')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary', ascending=False)
st.dataframe(cluster_means)

st.write("**Interpretation of Clusters:**")
st.write("- **High Value Customers:** Look for clusters with low Recency, high Frequency, and high Monetary values.")
st.write("- **Losing Customers:** Look for clusters with high Recency, low Frequency, and low Monetary values.")
st.write("- **New Customers:** Look for clusters with low Recency, but potentially lower Frequency/Monetary.")
st.write("- **Potential Loyalist:** Customers who have recently purchased, bought often, and spent a good amount.")

selected_cluster = st.selectbox('Select a Cluster to explore:', sorted(rfm['Cluster_KMeans'].unique()))

if selected_cluster is not None:
    st.write(f"Details for Cluster {selected_cluster}:")
    st.dataframe(rfm[rfm['Cluster_KMeans'] == selected_cluster].describe())
