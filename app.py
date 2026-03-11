import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Online Retail Customer Segmentation")

uploaded_file = st.file_uploader("Upload Online Retail Dataset")

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file, encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename columns if needed
    if "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "CustomerID"})

    if "Invoice" in df.columns:
        df = df.rename(columns={"Invoice": "InvoiceNo"})

    if "Price" in df.columns:
        df = df.rename(columns={"Price": "UnitPrice"})

    # Show columns (for debugging)
    st.write("Dataset Columns:", df.columns)

    # Remove missing customers
    df = df.dropna(subset=['CustomerID'])

    # Convert date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Remove negative quantity
    df = df[df['Quantity'] > 0]

    # Total price
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # RFM Calculation
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.subheader("RFM Table")
    st.dataframe(rfm.head())

    # Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    st.subheader("Customer Segments")
    st.write(rfm.groupby("Cluster").mean())

    # PCA for visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(rfm_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=rfm['Cluster'])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Customer Segmentation")

    st.pyplot(fig)
