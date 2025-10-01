import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(
    page_title="Pemetaan Tingkat Stres & Strategi Coping", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CSS Styling
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main {
        background: linear-gradient(135deg, #f5f2eb 0%, #ede5d8 100%);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #8b7355 0%, #a0916b 50%, #b8a082 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(139,115,85,0.3);
        text-align: center;
    }
    .main-header h1 {
        color: #f5f2eb;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f5f2eb;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f5f0 0%, #f0ead6 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #d4c4a8;
        box-shadow: 0 2px 12px rgba(139,115,85,0.1);
    }
    .metric-card h3 {
        color: #6b5b3d;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-card p {
        color: #8b7355;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #a99675 0%, #c2b28a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(139,115,85,0.25);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b7355 0%, #a0916b 100%);
        color: #f5f2eb;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139,115,85,0.35);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f8f5f0;
        border: 1px solid #d4c4a8;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        color: #6b5b3d;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b7355 0%, #a0916b 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(139,115,85,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("❌ File 'data.csv' tidak ditemukan.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error membaca 'data.csv': {e}")
        st.stop()

data = load_data()

if data is None or data.shape[0] == 0:
    st.error("❌ Dataset kosong.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌱 Pemetaan Tingkat Stres & Strategi Coping</h1>
    <p>Analisis Clustering untuk Memahami Pola Stres Mahasiswa</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Eksplorasi Data", "🎯 Clustering & Visualisasi"]) 

with tab1:
    st.markdown("### 📊 Dataset Mahasiswa")
    st.markdown("Berikut adalah overview dari data yang digunakan:")

    # hanya kolom numerik
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("👩‍🎓 Jumlah Data", len(data))
        with col2: st.metric("📊 Kolom Numerik", len(numeric_data.columns))
        with col3: st.metric("❌ Missing Values", numeric_data.isnull().sum().sum())

    st.dataframe(data, use_container_width=True, height=400)

with tab2:
    st.markdown("### 🎯 Konfigurasi Clustering")

    # Semester dimasukkan ke ignore_cols supaya tidak bisa dipilih
    ignore_cols = ["Timestamp", "Nama", "Jenis Kelamin", "Program Studi", "Tingkat Semester (angka saja, contoh: 3)"]
    numeric_cols = [c for c in data.columns if c not in ignore_cols and data[c].dtype in [np.int64, np.float64]]

    if len(numeric_cols) == 0:
        st.error("⚠️ Tidak ada kolom numerik untuk clustering.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_features = st.multiselect(
                "🎯 Pilih Fitur untuk Clustering (minimal 2):",
                options=numeric_cols,
                default=numeric_cols[:2],
            )
        with col2:
            max_k = min(10, max(2, len(data)))
            n_clusters = st.slider("🔢 Jumlah Cluster (k):", 2, max_k, min(3, max_k))

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            start_button = st.button("🚀 Mulai Analisis Clustering", use_container_width=True)

        if start_button:
            if len(selected_features) < 2:
                st.warning("⚠️ Pilih minimal 2 fitur numerik.")
            else:
                with st.status("🔄 Sedang memproses clustering...", expanded=False) as status:
                    valid_mask = ~data[selected_features].isnull().any(axis=1)
                    n_valid = int(valid_mask.sum())
                    n_dropped = int((~valid_mask).sum())

                    if n_valid == 0:
                        st.error("❌ Semua baris memiliki nilai kosong.")
                        st.stop()
                    elif n_valid < n_clusters:
                        st.error(f"❌ Data valid ({n_valid}) < jumlah cluster ({n_clusters}).")
                        st.stop()
                    else:
                        if n_dropped > 0:
                            st.warning(f"⚠️ {n_dropped} baris diabaikan (nilai kosong).")

                        # Clustering
                        from sklearn.decomposition import PCA
                        X = data.loc[valid_mask, selected_features].copy()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)

                        clustered_data = data.copy()
                        clustered_data['Cluster'] = pd.NA
                        clustered_data.loc[valid_mask, 'Cluster'] = clusters

                        # Tambahkan PCA 2D untuk visualisasi
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X_scaled)
                        clustered_data.loc[valid_mask, "PCA1"] = pca_result[:, 0]
                        clustered_data.loc[valid_mask, "PCA2"] = pca_result[:, 1]

                        status.update(label="✅ Clustering berhasil!", state="complete")

                # Hasil Clustering
                st.markdown("### 📊 Hasil Clustering")
                st.dataframe(clustered_data, use_container_width=True, height=400)

                # Visualisasi pakai PCA
                st.markdown("### 📈 Visualisasi Clustering Interaktif")
                plot_df = clustered_data.loc[valid_mask].copy()
                plot_df['Cluster'] = plot_df['Cluster'].astype(str)

                hover_cols = ["Nama", "Jenis Kelamin", "Program Studi", "Tingkat Semester (angka saja, contoh: 3)"]

                bright_colors = ['#e6194b', '#3cb44b', '#4363d8',
                                 '#f58231', '#911eb4', '#46f0f0',
                                 '#f032e6', "#d2bb26",'#bcf60c', '#fabebe']

                fig = px.scatter(
                    plot_df, x="PCA1", y="PCA2",
                    color='Cluster', hover_data=hover_cols,
                    title='Hasil Clustering K-Means (PCA 2D)',
                    color_discrete_sequence=bright_colors
                )
                fig.update_layout(
                    plot_bgcolor='#f8f5f0', paper_bgcolor='#ffffff',
                    font=dict(family="Inter, sans-serif", color='#6b5b3d'),
                    title_font=dict(size=18, color='#6b5b3d'),
                    legend=dict(bgcolor='rgba(248,245,240,0.8)', bordercolor='#d4c4a8', borderwidth=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Statistik per Cluster
                st.markdown("### 📊 Statistik Rata-rata per Cluster")
                cluster_summary = plot_df.groupby('Cluster')[selected_features].mean().round(2)
                st.dataframe(cluster_summary)

                # Export
                st.markdown("### 💾 Export Hasil")
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        png_bytes = fig.to_image(format='png')
                        st.download_button("🖼️ Download Visualisasi (PNG)", png_bytes,
                                           file_name="clustering_result.png", mime="image/png",
                                           use_container_width=True)
                    except Exception:
                        st.info("💡 Install 'kaleido' untuk ekspor PNG: pip install kaleido")

                with col2:
                    csv_bytes = clustered_data.to_csv(index=False).encode('utf-8')
                    st.download_button("📊 Download Hasil (CSV)", csv_bytes,
                                       file_name="clustering_results.csv", mime="text/csv",
                                       use_container_width=True)
