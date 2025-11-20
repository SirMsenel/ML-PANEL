# pages/model_oneri.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, pearsonr, f_oneway

st.set_page_config(page_title="Model Öneri Analizi", page_icon="🧭", layout="wide")
st.title("🧭 Model Seçimi İçin Veri Analizi")
st.markdown("---")

# ✅ Veri kontrolü
if "uploaded_df" not in st.session_state:
    st.warning("Önce veri yükleyin!")
    st.stop()

df = st.session_state["uploaded_df"].copy()

st.write("Bu sayfa model kurmadan önce veriyi analiz ederek **hangi modelin daha uygun olduğunu önerir.**")

# ------------------ Hedef Seçimi ------------------

# 🎯 Hedef değişken sadece sayısal olmalı
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

target = st.selectbox("🎯 Tahmin Edilecek Değişken (Y) — (Sadece Sayısal)", [None] + numeric_cols)

if target is None:
    st.warning("Lütfen tahmin edilecek **sayısal** bir hedef değişken seçin.")
    st.stop()

if not np.issubdtype(df[target].dtype, np.number):
    st.error("⚠️ Hedef değişken sayısal olmalıdır. Lütfen başka bir değişken seçin.")
    st.stop()

# Kullanıcıya bağımsız değişken seçme hakkı
candidate_features = [c for c in df.columns if c != target]
selected_features = st.multiselect("🔧 Analize dahil edilecek bağımsız değişkenler", candidate_features, default=candidate_features)

if not selected_features:
    st.warning("En az bir değişken seçmelisiniz.")
    st.stop()


# Ayrım: sayısal vs kategorik
num_feats = [c for c in selected_features if np.issubdtype(df[c].dtype, np.number)]
cat_feats = [c for c in selected_features if c not in num_feats]

# ------------------ Etki Hesaplama ------------------

def eta_squared(groups):
    total = pd.concat(groups)
    grand_mean = total.mean()
    ss_between = sum([len(g) * (g.mean() - grand_mean) ** 2 for g in groups])
    ss_total = sum((total - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else 0

results = []

# Sayısal değişkenlerde Pearson R
for col in num_feats:
    r, _ = pearsonr(df[col], df[target])
    results.append([col, "Sayısal", abs(r)])

# Kategorik değişkenlerde Eta-Squared
for col in cat_feats:
    groups = [df[df[col] == val][target].dropna() for val in df[col].dropna().unique()]
    if len(groups) > 1:
        eta = eta_squared(groups)
        results.append([col, "Kategorik", eta])

effect_df = pd.DataFrame(results, columns=["Değişken", "Tür", "Etki Gücü (0-1)"]).sort_values("Etki Gücü (0-1)", ascending=False)

st.subheader("📌 Bağımsız Değişkenlerin Hedef Üzerindeki Etkisi")
st.dataframe(effect_df, use_container_width=True)

st.info("""
**Yorumlama Rehberi:**
- **0.00 - 0.20:** Zayıf ilişki  
- **0.20 - 0.50:** Orta ilişki  
- **0.50+** : Güçlü ilişki  

• Sayısal değişkenlerde bu ölçü *Pearson Korelasyon (|R|)* değeridir.  
• Kategorik değişkenlerde bu ölçü *Eta-Squared* değeridir (ANOVA etkisi).  
""")

# ------------------ Model Yönlendirme ------------------
strong = effect_df[effect_df["Etki Gücü (0-1)"] >= 0.50]
medium = effect_df[(effect_df["Etki Gücü (0-1)"] >= 0.20) & (effect_df["Etki Gücü (0-1)"] < 0.50)]

st.markdown("---")
st.subheader("🧭 Model Önerisi")

recommendations = []

if len(strong) > 0 and (strong["Tür"] == "Sayısal").any():
    recommendations += ["Linear Regression", "XGBoost"]

if len(strong) > 0 and (strong["Tür"] == "Kategorik").any():
    recommendations += ["Random Forest", "CatBoost"]

if len(strong) == 0 and len(medium) > 0:
    recommendations += ["Random Forest", "Decision Tree"]

if len(recommendations) == 0:
    recommendations = ["Daha fazla veri veya özellik gerekebilir"]

st.success(f"**Önerilen Model(ler):** {', '.join(dict.fromkeys(recommendations))}")

# ------------------ Dağılım İncelemesi ------------------
st.markdown("---")
st.subheader("📊 Hedef Değişken Dağılımı")

fig, ax = plt.subplots(figsize=(6,3))
sns.histplot(df[target], kde=True, ax=ax, color="orange")
st.pyplot(fig)

skew_val = skew(df[target].dropna())
st.write(f"**Çarpıklık (Skew):** {round(skew_val, 2)}")

if abs(skew_val) < 0.5:
    st.success("✅ Hedef değişken yaklaşık normal dağılıyor → Doğrusal modeller çalışabilir.")
elif abs(skew_val) < 1.5:
    st.info("ℹ️ Biraz çarpık → Her iki model tipi de denenebilir.")
else:
    st.warning("⚠️ Çok çarpık dağılım → Tree tabanlı modeller daha uygundur.")

# ------------------ GRAFİKLER ------------------
st.markdown("---")
st.subheader("📊 Görsel İlişki Analizi")

# Sayısal ilişkiler: Scatter + Pairplot
num_for_plot = [c for c in num_feats if c != target]

if len(num_for_plot) > 0:
    st.markdown("### 🔹 Sayısal Değişkenlerde İlişki (Pairplot)")

    selected_plot_nums = st.multiselect("Grafikte gösterilecek sayısal değişkenleri seç", num_for_plot, default=num_for_plot[:3])

    if selected_plot_nums:
        try:
            fig = sns.pairplot(df[[target] + selected_plot_nums], kind="reg", diag_kind="kde")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Pairplot oluşturulurken hata: {e}")

# Kategorik: Boxplot
if len(cat_feats) > 0:
    st.markdown("### 🔹 Kategorik Değişkenlerde Hedef Dağılımı (Boxplot)")

    selected_cat_plot = st.selectbox("Boxplot için kategorik sütun seçin:", cat_feats)

    fig2, ax2 = plt.subplots(figsize=(7,4))
    sns.boxplot(x=df[selected_cat_plot], y=df[target], ax=ax2, palette="Set2")
    ax2.set_xlabel(selected_cat_plot)
    ax2.set_ylabel(target)
    st.pyplot(fig2)
