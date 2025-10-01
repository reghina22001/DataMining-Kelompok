import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pemetaan Tingkat Stres & Strategi Coping", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="main-header">
    <h1>🌱 Pemetaan Tingkat Stres & Strategi Coping</h1>
    <p>Analisis K-Means Clustering untuk Memahami Pola Stres Mahasiswa</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Eksplorasi Data", "🎯 Clustering & Visualisasi"]) 

with tab1:
    st.markdown("### 📊 Dataset Mahasiswa")
    
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 0:
        col1, col2, col3= st.columns(3)
        with col1: st.metric("👩‍🎓 Jumlah Data", len(data))
        with col2: st.metric("📊 Kolom Numerik", len(numeric_data.columns))
        with col3: st.metric("❌ Missing Values", numeric_data.isnull().sum().sum())

    st.dataframe(data, use_container_width=True, height=400)
    
    st.markdown("### 📈 Distribusi Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribusi Jenis Kelamin")
        if 'Jenis Kelamin' in data.columns:
            gender_dist = data['Jenis Kelamin'].value_counts()
            fig_gender = px.pie(
                values=gender_dist.values, 
                names=gender_dist.index, 
                color_discrete_sequence=['#8b7355', '#c2b28a', '#a0916b']
            )
            fig_gender.update_layout(
                plot_bgcolor='#f8f5f0',
                paper_bgcolor='#ffffff',
                font=dict(family="Inter, sans-serif", color='#6b5b3d')
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.warning("Kolom 'Jenis Kelamin' tidak ditemukan")
    
    with col2:
        st.markdown("#### Distribusi Semester")
        semester_col = "Tingkat Semester (angka saja, contoh: 3)"
        if semester_col in data.columns:
            semester_dist = data[semester_col].value_counts().sort_index()
            fig_semester = px.bar(
                x=semester_dist.index, 
                y=semester_dist.values,
                labels={'x': 'Semester', 'y': 'Jumlah Mahasiswa'},
                color=semester_dist.values,
                color_continuous_scale='Viridis'
            )
            fig_semester.update_layout(
                plot_bgcolor='#f8f5f0',
                paper_bgcolor='#ffffff',
                font=dict(family="Inter, sans-serif", color='#6b5b3d'),
                showlegend=False,
                xaxis_title="Semester",
                yaxis_title="Jumlah Mahasiswa",
                xaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=2
                )
            )
            st.plotly_chart(fig_semester, use_container_width=True)
        else:
            st.warning(f"Kolom '{semester_col}' tidak ditemukan")

with tab2:
    st.markdown("### 🎯 Konfigurasi K-Means Clustering")

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
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:2],
                help="Pilih variabel yang akan digunakan untuk mengelompokkan mahasiswa"
            )
        with col2:
            max_k = min(10, max(2, len(data)))
            n_clusters = st.slider("🔢 Jumlah Cluster (k):", 2, max_k, min(3, max_k),
                                  help="Jumlah kelompok yang ingin dibentuk")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            start_button = st.button("🚀 Mulai Analisis K-Means", use_container_width=True)

        if start_button:
            if len(selected_features) < 2:
                st.warning("⚠️ Pilih minimal 2 fitur numerik.")
            else:
                with st.spinner("🔄 Memproses K-Means Clustering..."):
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

                        X = data.loc[valid_mask, selected_features].copy()
                        
                        # Standardisasi (penting untuk K-Means!)
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # K-Means Clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
                        clusters = kmeans.fit_predict(X_scaled)

                        # Evaluasi clustering
                        silhouette = silhouette_score(X_scaled, clusters)
                        davies_bouldin = davies_bouldin_score(X_scaled, clusters)
                        inertia = kmeans.inertia_

                        clustered_data = data.copy()
                        clustered_data['Cluster'] = pd.NA
                        clustered_data.loc[valid_mask, 'Cluster'] = clusters

                        # PCA untuk visualisasi
                        pca = PCA(n_components=2)
                        pca_result = pca.fit_transform(X_scaled)
                        clustered_data.loc[valid_mask, "PCA1"] = pca_result[:, 0]
                        clustered_data.loc[valid_mask, "PCA2"] = pca_result[:, 1]
                        
                        explained_var = pca.explained_variance_ratio_

                        st.success("✅ Clustering berhasil!")

                # Metrik Evaluasi
                st.markdown("### 📊 Metrik Evaluasi K-Means")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}", 
                             help="Semakin tinggi (mendekati 1), semakin baik clustering. Range: -1 hingga 1")
                with col2:
                    st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}",
                             help="Semakin rendah, semakin baik clustering")
                with col3:
                    st.metric("Inertia (WCSS)", f"{inertia:.2f}",
                             help="Sum of squared distances ke centroid terdekat")

                # Hasil Clustering
                st.markdown("### 📊 Hasil Clustering")
                st.dataframe(clustered_data, use_container_width=True, height=400)

                # Visualisasi PCA
                st.markdown("### 📈 Visualisasi Clustering (PCA 2D)")
                st.info(f"📌 PCA Component 1 menjelaskan {explained_var[0]*100:.1f}% varians, Component 2 menjelaskan {explained_var[1]*100:.1f}% varians")
                
                plot_df = clustered_data.loc[valid_mask].copy()
                plot_df['Cluster'] = 'Cluster ' + plot_df['Cluster'].astype(str)

                hover_cols = ["Nama", "Jenis Kelamin", "Program Studi"]

                bright_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', 
                               '#46f0f0', '#f032e6', '#d2bb26', '#bcf60c', '#fabebe']

                fig = px.scatter(
                    plot_df, x="PCA1", y="PCA2",
                    color='Cluster', hover_data=hover_cols,
                    title='K-Means Clustering Results (PCA 2D Projection)',
                    color_discrete_sequence=bright_colors,
                    size_max=10
                )
                
                # Tambahkan centroid
                centroids_pca = pca.transform(kmeans.cluster_centers_)
                for i, centroid in enumerate(centroids_pca):
                    fig.add_trace(go.Scatter(
                        x=[centroid[0]], y=[centroid[1]],
                        mode='markers',
                        marker=dict(size=20, color='black', symbol='x', line=dict(width=2, color='white')),
                        name=f'Centroid {i}',
                        showlegend=True
                    ))
                
                fig.update_layout(
                    plot_bgcolor='#f8f5f0', 
                    paper_bgcolor='#ffffff',
                    font=dict(family="Inter, sans-serif", color='#6b5b3d'),
                    title_font=dict(size=18, color='#6b5b3d'),
                    legend=dict(bgcolor='rgba(248,245,240,0.8)', bordercolor='#d4c4a8', borderwidth=1),
                    xaxis_title=f"PC1 ({explained_var[0]*100:.1f}% variance)",
                    yaxis_title=f"PC2 ({explained_var[1]*100:.1f}% variance)"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Export
                st.markdown("### 💾 Export Hasil")
                try:
                    png_bytes = fig.to_image(format='png', width=1200, height=800)
                    st.download_button(
                        "🖼️ Download Visualisasi (PNG)", 
                        png_bytes,
                        file_name="clustering_visualization.png", 
                        mime="image/png",
                        use_container_width=True
                    )
                except Exception as e:
                    st.info("💡 Install 'kaleido' untuk ekspor PNG: pip install kaleido")
                    st.error(f"Error: {e}")