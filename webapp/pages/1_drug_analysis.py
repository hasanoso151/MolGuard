"""
MolGuard — İlaç Analizi Sayfası / Drug Analysis Page
"""

import streamlit as st
import numpy as np

st.title("🔍 İlaç Analizi / Drug Analysis")

st.markdown("""
Bir molekül seçin ve güvenlik profilini görün.
Select a molecule and view its safety profile.
""")

# Molekül seçimi / Molecule selection
mol_idx = st.slider("Molekül İndeksi / Molecule Index", 0, 7830, 0)

col1, col2, col3 = st.columns(3)

# Simüle edilmiş sonuçlar / Simulated results
np.random.seed(mol_idx)
tox_score = np.random.uniform(0.1, 0.9)
efficacy = np.random.uniform(0.3, 0.95)
side_effects = np.random.randint(1, 15)

with col1:
    st.metric("🔴 Toksisite / Toxicity", f"{tox_score:.2%}",
              delta=f"{'Yüksek/High' if tox_score > 0.5 else 'Düşük/Low'}")

with col2:
    st.metric("🟢 Etkinlik / Efficacy", f"{efficacy:.2%}")

with col3:
    st.metric("🟡 Yan Etkiler / Side Effects", f"{side_effects} olası/possible")

# Risk değerlendirmesi / Risk assessment
st.markdown("---")
st.subheader("👤 Hasta Profili Risk Analizi / Patient Risk Analysis")

age = st.slider("Yaş / Age", 18, 90, 45)
conditions = st.multiselect(
    "Mevcut Durumlar / Existing Conditions",
    ["Diyabet / Diabetes", "Hipertansiyon / Hypertension",
     "Böbrek Hastalığı / Kidney Disease", "Karaciğer / Liver Disease"]
)

risk_mult = 1.0 + (age - 45) * 0.01 + len(conditions) * 0.15
adjusted_risk = min(tox_score * risk_mult, 1.0)

if adjusted_risk < 0.3:
    st.success(f"🟢 Düşük Risk / Low Risk: {adjusted_risk:.2%}")
elif adjusted_risk < 0.6:
    st.warning(f"🟡 Orta Risk / Medium Risk: {adjusted_risk:.2%}")
else:
    st.error(f"🔴 Yüksek Risk / High Risk: {adjusted_risk:.2%}")
