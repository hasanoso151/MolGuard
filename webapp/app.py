"""
MolGuard — Web Uygulaması / Web Application
Streamlit ile ilaç güvenlik analiz arayüzü
Drug safety analysis interface with Streamlit
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os

# Proje kök dizinini ekle / Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="🧬 MolGuard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ana Sayfa / Home Page
st.title("🧬 MolGuard")
st.subheader("Akıllı İlaç Güvenlik Analiz Sistemi | Intelligent Drug Safety Analysis System")

st.markdown("---")

# Proje açıklaması / Project description
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Ne Yapar? / What Does It Do?

    MolGuard, ilaçların moleküler yapılarını **Grafik Sinir Ağları (GNN)**
    kullanarak analiz eder ve güvenlik tahminleri yapar.

    MolGuard analyzes drug molecular structures using **Graph Neural Networks**
    and predicts safety profiles.
    """)

with col2:
    st.markdown("""
    ### ✨ Özellikler / Features

    - 🔴 **Toksisite Tahmini** / Toxicity Prediction
    - 🟡 **Yan Etki Tahmini** / Side Effect Prediction
    - 🟢 **Etkinlik Tahmini** / Efficacy Prediction
    - ⚠️ **İlaç Etkileşimi** / Drug Interaction
    - 💊 **Güvenli Alternatif** / Safe Alternative
    - 🧠 **Açıklanabilirlik** / Explainability
    """)

st.markdown("---")

# Sonuçlar / Results
st.subheader("📊 Model Sonuçları / Model Results")

results_df = pd.DataFrame({
    'Model': ['GCN', 'GAT', 'GIN', 'Multi-Task GNN'],
    'Test AUC': [0.7479, 0.7315, 0.6819, 0.7431],
    'Açıklama / Description': [
        'Temel GNN / Baseline GNN',
        'Dikkat mekanizmalı / With attention',
        'En güçlü ayırt edici / Most expressive',
        'Çok görevli / Multi-task (4 datasets)'
    ]
})

st.dataframe(results_df, use_container_width=True, hide_index=True)

# Sidebar
st.sidebar.title("📌 Navigasyon / Navigation")
st.sidebar.markdown("""
- 🏠 Ana Sayfa / Home
- 🔍 İlaç Analizi / Drug Analysis
- ⚔️ İlaç Karşılaştırma / Drug Compare
- ⚠️ Etkileşim Kontrolü / Interaction Check
- 💊 Güvenli Alternatif / Safe Alternative
- 🧠 Keşif / Explore
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**Öğrenci / Student:** Hasan (230212925)
**Üniversite:** Osim Teknik Üniversitesi
**Ders:** Derin Öğrenme / Deep Learning
""")
