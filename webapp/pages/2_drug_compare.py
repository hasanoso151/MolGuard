"""
MolGuard — İlaç Karşılaştırma / Drug Comparison
"""

import streamlit as st
import numpy as np
import pandas as pd

st.title("⚔️ İlaç Karşılaştırma / Drug Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💊 İlaç A / Drug A")
    mol_a = st.number_input("Molekül A İndeksi", 0, 7830, 0)

with col2:
    st.subheader("💊 İlaç B / Drug B")
    mol_b = st.number_input("Molekül B İndeksi", 0, 7830, 100)

if st.button("🔍 Karşılaştır / Compare"):
    np.random.seed(mol_a)
    tox_a, eff_a, se_a = np.random.uniform(0.1, 0.9), np.random.uniform(0.3, 0.95), np.random.randint(1, 15)

    np.random.seed(mol_b)
    tox_b, eff_b, se_b = np.random.uniform(0.1, 0.9), np.random.uniform(0.3, 0.95), np.random.randint(1, 15)

    comp = pd.DataFrame({
        'Metrik / Metric': ['Toksisite / Toxicity', 'Etkinlik / Efficacy', 'Yan Etkiler / Side Effects'],
        f'İlaç A #{mol_a}': [f'{tox_a:.2%}', f'{eff_a:.2%}', se_a],
        f'İlaç B #{mol_b}': [f'{tox_b:.2%}', f'{eff_b:.2%}', se_b],
    })

    st.dataframe(comp, use_container_width=True, hide_index=True)

    # Öneri / Recommendation
    if tox_a < tox_b:
        st.success(f"💡 Öneri: İlaç A #{mol_a} daha güvenli / Drug A is safer")
    else:
        st.success(f"💡 Öneri: İlaç B #{mol_b} daha güvenli / Drug B is safer")
