"""
MolGuard — Keşif Sayfası / Explore Page
Gömme uzayı ve model sonuçları
Embedding space and model results
"""

import streamlit as st
import pandas as pd

st.title("🧠 Keşif / Explore")

st.markdown("""
Bu sayfa model sonuçlarını ve gömme uzayını keşfetmenizi sağlar.
This page lets you explore model results and embedding space.
""")

# Model karşılaştırma / Model comparison
st.subheader("📊 Model Karşılaştırma / Model Comparison")

results = pd.DataFrame({
    'Model': ['GCN', 'GAT', 'GIN', 'Multi-Task'],
    'Valid AUC': [0.7883, 0.7711, 0.7171, '-'],
    'Test AUC': [0.7479, 0.7315, 0.6819, 0.7431],
})
st.dataframe(results, use_container_width=True, hide_index=True)

# Multi-task sonuçları / Multi-task results
st.subheader("🧬 Multi-Task Sonuçları / Results")

mt_results = pd.DataFrame({
    'Görev / Task': ['Toxicity', 'Side Effects', 'Efficacy', 'Clinical'],
    'Veri Seti / Dataset': ['Tox21', 'SIDER', 'BBBP', 'ClinTox'],
    'Test AUC': [0.7431, 0.6415, 0.6883, 0.8100],
})
st.dataframe(mt_results, use_container_width=True, hide_index=True)

# Görselleştirmeler / Visualizations
st.subheader("🗺️ Görselleştirmeler / Visualizations")
st.info("Eğitim eğrileri, t-SNE gömme haritaları ve GNNExplainer sonuçları "
        "assets/results/ klasöründe bulunabilir. / Training curves, t-SNE maps, "
        "and GNNExplainer results can be found in assets/results/")
