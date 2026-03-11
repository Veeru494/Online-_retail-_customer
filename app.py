import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Online Retail Customer Segmentation")

# Upload dataset
uploaded_file = st.file_uploader("Upload Online Retail Dataset")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, encoding='latin1')

    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['Quantity'] > 0]

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency','Frequency','Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.write("RFM Table with Clusters")
    st.dataframe(rfm.head())

    # PCA visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(rfm_scaled)

    plt.scatter(pca_data[:,0], pca_data[:,1], c=rfm['Cluster'])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Customer Segmentation")
    
    st.pyplot(plt)
