# pages/model_rapor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Model Raporu", page_icon="📊", layout="wide")
st.title("📊 Model Performans Raporu")
st.markdown("---")

# ✅ Model kontrolü
required_keys = ["model", "test_predictions", "test_truth", "X_columns", "X_full", "uploaded_df"]
for k in required_keys:
    if k not in st.session_state:
        st.error(f"'{k}' bilgisi eksik. Lütfen **Model Kur** sayfasında modeli yeniden eğitin.")
        st.stop()

model       = st.session_state["model"]
preds       = st.session_state["test_predictions"]
y_test      = st.session_state["test_truth"]
X_columns   = st.session_state["X_columns"]
X_full      = st.session_state["X_full"]
uploaded_df = st.session_state["uploaded_df"]

# ------------------ Performans Metrikleri ------------------
st.subheader("📌 Performans Metrikleri")

r2   = r2_score(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds, squared=False)

# Renk belirleme (R²'ye göre)
if r2 < 0.3:
    color = "#ff4b4b"  # Kırmızı
elif r2 < 0.7:
    color = "#ffa534"  # Turuncu
else:
    color = "#4bb543"  # Yeşil

# Daha küçük kutular
col1, col2, col3 = st.columns(3)

box_style = """
    background-color: #ffffff;
    padding: 12px;
    border-radius: 8px;
    border-left: 6px solid {color};
    text-align: center;
"""

with col1:
    st.markdown(f"""
    <div style="{box_style.format(color=color)}">
        <p style="margin:0; font-size:14px; color:#000;">R²</p>
        <p style="margin:0; font-size:22px; font-weight:bold; color:#000;">{r2:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="{box_style.format(color='#6fa8dc')}">
        <p style="margin:0; font-size:14px; color:#000;">MAE</p>
        <p style="margin:0; font-size:22px; font-weight:bold; color:#000;">{mae:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="{box_style.format(color='#8e7cc3')}">
        <p style="margin:0; font-size:14px; color:#000;">RMSE</p>
        <p style="margin:0; font-size:22px; font-weight:bold; color:#000;">{rmse:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

# 🔥 BURASI EKLENDİ → kutucuklar ve yorum arasına boşluk
st.markdown("<br>", unsafe_allow_html=True)

# Yorum
if r2 < 0.3:
    st.error("📉 Model hedef değişkeni **zayıf açıklıyor**.")
elif r2 < 0.7:
    st.warning("⚖️ Model **orta düzeyde açıklıyor**. Tuning yapılabilir.")
else:
    st.success("🚀 Model **yüksek başarı gösteriyor!** ✅")

st.markdown("---")



# ------------------ Hata Analizi ------------------
st.subheader("📦 Hata Analizi")

# Hata ve yüzdesel hata (MAPE tarzı)
residuals = y_test - preds
percentage_error = (residuals / y_test.replace(0, np.nan)) * 100  # 0 bölen önlenir
percentage_error = percentage_error.fillna(0)  # NaN -> 0

report_df = pd.DataFrame({
    "Gerçek": y_test,
    "Tahmin": preds,
    "Hata": residuals,
    "Hata (%)": percentage_error.round(2)   # ✅ Yüzdesel hata eklendi
})

top_k = st.slider("Gösterilecek kayıt sayısı", 5, len(report_df), 10)

# Hatası en yüksek olanları sırala
sorted_report = report_df.reindex(residuals.abs().sort_values(ascending=False).index)

st.dataframe(sorted_report.head(top_k), use_container_width=True)

# Plotly kullanılacaksa importlar:
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.markdown("### 📈 Gerçek - Tahmin - Hata Çizgi Grafiği (Çift Eksen)")

line_df = pd.DataFrame({
    "Gerçek": y_test.values,
    "Tahmin": preds,
    "Hata": (y_test.values - preds)
})

# 👇 Çift y-ekseni grafiği oluştur
fig = make_subplots(specs=[[{"secondary_y": True}]])

# --- Sol eksen: Gerçek ---
fig.add_trace(
    go.Scatter(
        y=line_df["Gerçek"], 
        mode='lines', 
        name='Gerçek',
        line=dict(width=2, color="royalblue")
    ),
    secondary_y=False
)

# --- Sol eksen: Tahmin ---
fig.add_trace(
    go.Scatter(
        y=line_df["Tahmin"], 
        mode='lines', 
        name='Tahmin',
        line=dict(width=2, color="darkorange")
    ),
    secondary_y=False
)

# --- Sağ eksen: Hata ---
fig.add_trace(
    go.Scatter(
        y=line_df["Hata"], 
        mode='lines', 
        name='Hata (Gerçek - Tahmin)',
        line=dict(width=1.5, color="red", dash="dot")
    ),
    secondary_y=True
)

fig.update_layout(
    height=370,
    xaxis_title="Gözlem Index",
)

fig.update_yaxes(title_text="Gerçek / Tahmin", secondary_y=False)
fig.update_yaxes(title_text="Hata", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

# 🎯 Modelde Yer Alan Değişkenlerin Etkisi
st.markdown("---")
st.subheader("🎯 Modelde Yer Alan Değişkenlerin Etkisi (Gerçek Etki Değerleri)")

if hasattr(model, "coef_"):
    effect_df = pd.DataFrame({"Değişken": X_columns, "Etki": model.coef_})
else:
    effect_df = pd.DataFrame({"Değişken": X_columns, "Etki": model.feature_importances_})

# Etkileri büyükten küçüğe sırala
effect_df = effect_df.sort_values("Etki", ascending=False)

# Kullanıcının görmek istediği kaç değişken
top_n = st.slider("Gösterilecek değişken sayısı", 1, len(effect_df), min(10, len(effect_df)))

# Etki Tablosu
st.dataframe(effect_df.head(top_n), use_container_width=True)

# 🎨 Grafik (Etki değerleri NORMAL - normalize değil!)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=effect_df.head(top_n), x="Etki", y="Değişken", palette="Blues_r", ax=ax)
ax.set_title("Değişkenlerin Modele Gerçek Etkisi")
st.pyplot(fig)


# ---------------- SEÇİLİ DEĞİŞKEN BAZLI HATA DAVRANIŞI ----------------
st.markdown("### 📊 Seçili Değişken Bazlı Hata Davranışı")

selected_feat = st.selectbox("Değişken seç:", effect_df["Değişken"].tolist())

uploaded_df = st.session_state["uploaded_df"]
X_full = st.session_state.get("X_full", None)

if X_full is None:
    st.error("X matrisi bulunamadı. Lütfen Model Kur sayfasında modeli tekrar çalıştırın.")
    st.stop()

# Test setine karşılık gelen özellikler
X_test_dummy = X_full.loc[y_test.index]

analysis_df = pd.DataFrame({
    "Gerçek": y_test.values,
    "Tahmin": preds,
    "Hata": residuals
})

# Eğer değişken orijinal veri sütunu ise → direkt scatter
if selected_feat in uploaded_df.columns:

    analysis_df[selected_feat] = uploaded_df.loc[y_test.index, selected_feat]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=analysis_df, x=selected_feat, y="Gerçek", label="Gerçek", alpha=0.6, ax=ax)
    sns.scatterplot(data=analysis_df, x=selected_feat, y="Tahmin", label="Tahmin", alpha=0.6, ax=ax)
    ax.set_title(f"{selected_feat} - Gerçek vs Tahmin")
    st.pyplot(fig)

else:
    # Dummy değişken ise → kategoriyi otomatik belirle
    base_col = selected_feat.split("_")[0]   # örn: "Cinsiyet"
    category_value = selected_feat.split("_")[1]  # örn: "Kadın"

    # Test setindeki kategoriyi belirle
    analysis_df["Kategori"] = np.where(X_test_dummy[selected_feat] == 1, category_value, f"Diğer {base_col}")

    grouped = analysis_df.groupby("Kategori")["Hata"].mean().reset_index()

    st.write("**📦 Kategori Bazlı Ortalama Hata**")
    st.dataframe(grouped, use_container_width=True)

    # Otomatik yorum
    diff = grouped["Hata"].max() - grouped["Hata"].min()
    worst_group = grouped.loc[grouped["Hata"].idxmax(), "Kategori"]
    best_group = grouped.loc[grouped["Hata"].idxmin(), "Kategori"]

    if diff < (residuals.std() * 0.2):
        st.success("✅ Model bu değişkene göre dengeli tahmin yapıyor.")
    else:
        st.warning(f"⚠️ Model **{worst_group}** grubunda belirgin şekilde daha yüksek hata yapıyor.")
        st.info(f"💡 Bu durum **{base_col}** değişkeninin modele daha iyi temsil edilmesi gerektiğini gösterir.")



# ------------------ Scatter Plot ------------------
st.subheader("🎯 Gerçek vs Tahmin Dağılımı")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.scatter(y_test, preds, alpha=0.6)
ax1.set_xlabel("Gerçek Değer")
ax1.set_ylabel("Tahmin")
ax1.set_title("Scatter Plot")
st.pyplot(fig1)

# ------------------ Rezidü Analizi ------------------
st.subheader("📉 Hata (Rezidü) Dağılımı")
fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.histplot(residuals, kde=True, ax=ax2, color="purple")
ax2.set_title("Rezidü Histogram")
st.pyplot(fig2)

# Rezidü yorumu
if abs(np.mean(residuals)) < abs(np.std(residuals)) * 0.1:
    st.success("✅ Rezidülerin ortalaması 0'a yakın → Model yanlı değil.")
else:
    st.warning("⚠️ Rezidülerde yanlılık var → Model belirli alanlarda sistematik hata yapıyor olabilir.")


# ------------------ SHAP ------------------
st.markdown("---")
st.subheader("🧠 SHAP Model Açıklanabilirlik Analizi")

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = st.session_state["model"]
X_columns = st.session_state["X_columns"]
X_test_scaled = st.session_state["X_test_scaled"]   # Ölçekli test X
X_test_full = st.session_state["X_test_full"]       # Dummy genişletilmiş test X
y_test = st.session_state["test_truth"]
preds = st.session_state["test_predictions"]

# ✅ SHAP veri setini DataFrame'e çeviriyoruz
X_shap = pd.DataFrame(X_test_scaled, columns=X_columns)

# ✅ Model tipine göre explainer seçimi
model_name = model.__class__.__name__

if model_name in ["RandomForestRegressor", "DecisionTreeRegressor", "LGBMRegressor", "XGBRegressor"]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
else:
    explainer = shap.LinearExplainer(model, X_shap)
    shap_values = explainer.shap_values(X_shap)

st.success("✅ SHAP değerleri başarıyla hesaplandı!")

# ------------------ SHAP SUMMARY PLOT ------------------
st.write("### 🌍 Özelliklerin Model Tahminine Etki Dağılımı")

fig_summary = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_shap, feature_names=X_columns, show=False)
st.pyplot(fig_summary)
plt.clf()

# ------------------ SHAP DECISION PLOT ------------------
st.markdown("---")
st.subheader("🧭 SHAP Decision Plot (Tahmin Karar Akışı)")

if model_name in ["RandomForestRegressor", "DecisionTreeRegressor", "LGBMRegressor", "XGBRegressor"]:

    decision_index = st.slider("Karar yolunu incelemek istediğiniz gözlem", 
                               0, len(X_test_full)-1, 0)

    st.write(f"Seçilen Gözlem Tahmini: **{preds[decision_index]:.3f}**")
    st.write(f"Gerçek Değer: **{y_test.iloc[decision_index]:.3f}**")

    fig_decision, ax = plt.subplots(figsize=(10, 4))
    shap.decision_plot(explainer.expected_value, shap_values[decision_index], 
                       X_test_full.iloc[decision_index], show=False)
    st.pyplot(fig_decision)
    plt.clf()

else:
    st.info("ℹ️ Decision Plot sadece ağaç tabanlı modellerde kullanılabilir.")

# ------------------ SHAP FORCE PLOT ------------------
st.markdown("---")
st.subheader("🎯 Tek Gözlem İçin SHAP Force Plot (Neden Bu Tahmin?)")

force_index = st.slider("İncelenecek Gözlem (Index)", 0, len(X_test_full)-1, 0)

st.write(f"**Gerçek Değer:** {y_test.iloc[force_index]:.3f}")
st.write(f"**Tahmin:** {preds[force_index]:.3f}")

explanation = shap.Explanation(
    values = shap_values[force_index],
    base_values = explainer.expected_value,
    data = X_test_full.iloc[force_index]
)

shap.plots.force(
    explanation.base_values,
    explanation.values,
    explanation.data,
    matplotlib=True,
    show=False
)

fig_force = plt.gcf()
fig_force.set_size_inches(10, 2.6)
st.pyplot(fig_force)
plt.clf()

st.caption("""
Bu grafik seçilen gözlemin tahminine hangi değişkenlerin katkı sağladığını gösterir:
- 🔵 Pozitif → Tahmini **YÜKSELTİR**
- 🔴 Negatif → Tahmini **DÜŞÜRÜR**
""")

