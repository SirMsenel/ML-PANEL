# pages/predict.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Tahmin", page_icon="🔮", layout="wide")
st.title("🔮 Tahmin (Predict)")
st.markdown("---")

# ========== ÖN KONTROL ==========
required = ["model", "X_columns", "uploaded_df", "features"]
missing = [k for k in required if k not in st.session_state]
if missing:
    st.error("Önce **Model Kur** sayfasında model eğitip kaydediniz.")
    st.stop()

model       = st.session_state["model"]
X_columns   = st.session_state["X_columns"]
uploaded_df = st.session_state["uploaded_df"]
features    = st.session_state["features"]

# Ölçekleme var mı?
scaled_used = ("scaler_mean_" in st.session_state) and ("scaler_scale_" in st.session_state)
scaler_mean = st.session_state.get("scaler_mean_", None)
scaler_scale= st.session_state.get("scaler_scale_", None)

# ========== Eğitimdeki Gibi Ön İşleme ==========
def preprocess_like_training(df_new):
    # Dummy encode
    X_new = pd.get_dummies(df_new, drop_first=False)

    # Eğitim kolonlarına hizala
    X_new = X_new.reindex(columns=X_columns, fill_value=0)

    # Ölçekleme
    if scaled_used:
        X_new = (X_new - scaler_mean) / scaler_scale

    return X_new

# ========== SEKME OLUŞTUR ==========
tab_csv, tab_form = st.tabs(["📦 Toplu Tahmin (CSV)", "🧍 Tek Kayıt Tahmini"])

# --------------------------------------------------------------------
# 📦 TOPLU TAHMİN (CSV)
# --------------------------------------------------------------------
with tab_csv:
    st.subheader("📦 Toplu Tahmin (CSV / Excel)")

    file = st.file_uploader("CSV veya Excel yükleyin (hedef sütun Y **olmasın**)", type=["csv", "xlsx"])

    if file is not None:
        try:
            df_new = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        except Exception as e:
            st.error(f"❌ Dosya okunamadı: {e}")
            st.stop()

        st.write("**📌 Yüklenen veri (ilk 5 satır):**")
        st.dataframe(df_new.head(), use_container_width=True)

        # --- Modelin beklediği dummy yapısına getir ---
        df_dummy = pd.get_dummies(df_new, drop_first=False)

        missing_cols = set(X_columns) - set(df_dummy.columns)
        if missing_cols:
            st.warning(f"⚠️ Modelin beklediği {len(missing_cols)} kolon eksik:")
            st.write(list(missing_cols))

            if st.checkbox("Eksik kolonları otomatik olarak **0** ile ekle", value=False):
                df_dummy = df_dummy.reindex(columns=X_columns, fill_value=0)
            else:
                st.stop()
        else:
            df_dummy = df_dummy.reindex(columns=X_columns, fill_value=0)

        # --- Ölçekleme ---
        if scaled_used:
            df_dummy = (df_dummy - scaler_mean) / scaler_scale

        # --- Tahmin ---
        preds = model.predict(df_dummy)

        # --- ✅ SADECE ORİJİNAL VERİ + PREDİCTION ---
        result = df_new.copy()   # sadece yüklenen veri
        result["Prediction"] = preds

        st.success("✅ Tahminler hazır")
        st.write("**📊 Sonuç (ilk 30 satır):**")
        st.dataframe(result.head(30), use_container_width=True)

        st.download_button(
            "💾 Tahminleri indir (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            file_name="tahmin_sonuclari.csv",
            mime="text/csv"
        )

# --------------------------------------------------------------------
# 🧍 TEK KAYIT TAHMİNİ
# --------------------------------------------------------------------
with tab_form:
    st.subheader("🧍 Tek Kayıt Tahmini (Form)")
    st.caption("Sadece modelde kullanılan bağımsız değişkenler gösterilir.")

    col_left, col_right = st.columns(2)
    inputs = {}

    from datetime import datetime

    for i, col in enumerate(features):
        slot = col_left if i % 2 == 0 else col_right

        col_data = uploaded_df[col].dropna()

        # 1) TARİH ALGILAMA
        if np.issubdtype(col_data.dtype, np.datetime64):
            default_date = col_data.iloc[0] if len(col_data) > 0 else pd.Timestamp.today()
            inputs[col] = slot.date_input(col, default_date.date())

        # 2) SAAT ALGILAMA (string HH:MM)
        elif col_data.astype(str).str.match(r"^\d{2}:\d{2}(:\d{2})?$").all():
            try:
                default_time = pd.to_datetime(col_data.iloc[0]).time()
            except:
                default_time = datetime.now().time()
            inputs[col] = slot.time_input(col, default_time)

        # 3) SAYISAL → SERBEST number_input (KESİKLİ DÖNÜŞÜM MODELDE HALA YAPILIYOR)
        elif pd.api.types.is_numeric_dtype(col_data.dtype):
            mn = float(col_data.min())
            mx = float(col_data.max())
            md = float(col_data.median())
            inputs[col] = slot.number_input(col, value=md)

        # 4) KATEGORİK → selectbox
        else:
            uniques = sorted(col_data.astype(str).unique())
            inputs[col] = slot.selectbox(col, uniques)

    if st.button("🔮 Tahmin Et", key="predict_single"):

        df_single = pd.DataFrame([inputs])

        # Tarihleri gerçek datetime yap
        for col in df_single.columns:
            if np.issubdtype(uploaded_df[col].dtype, np.datetime64):
                df_single[col] = pd.to_datetime(df_single[col])

        # Saat string'lerini modele uygun numerik saate çevir (opsiyonel fakat mantıklı)
        for col in df_single.columns:
            if df_single[col].astype(str).str.match(r"^\d{2}:\d{2}(:\d{2})?$").all():
                df_single[col] = pd.to_timedelta(df_single[col].astype(str)).dt.total_seconds() / 3600

        X_single = preprocess_like_training(df_single)
        pred = model.predict(X_single)[0]

        st.markdown(f"""
        <div style="padding:18px;border-radius:10px;background:#f8f9fe;border-left:6px solid #6c63ff;
        font-size:18px;color:black;">
        🔮 <b>Tahmin Sonucu:</b> {pred:.4f}
        </div>
        """, unsafe_allow_html=True)

        # ================= SHAP Mini Açıklaması =================
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        # SHAP veri uyumu
        X_shap_single = X_single.copy()

        model_name = model.__class__.__name__
        if model_name in ["RandomForestRegressor", "DecisionTreeRegressor", "LGBMRegressor", "XGBRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap_single)
        else:
            explainer = shap.LinearExplainer(model, X_shap_single)
            shap_values = explainer.shap_values(X_shap_single)

        # Tek satır SHAP değerleri
        shap_row = np.array(shap_values)[0]

        # Küçük bar plot
        st.markdown("#### 🔍 Bu tahmini en çok etkileyen değişkenler")
        shap_df = pd.DataFrame({
            "Değişken": X_single.columns,
            "Etkisi": shap_row
        }).sort_values("Etkisi", key=abs, ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(shap_df["Değişken"], shap_df["Etkisi"], color=["#6c63ff" if v>0 else "#ff6b6b" for v in shap_df["Etkisi"]])
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Tahmine Katkı Payları (SHAP)")
        st.pyplot(fig)

        # İndirilebilir sonuç
        result = df_single.copy()
        result["Prediction"] = pred
        st.dataframe(result, use_container_width=True)

        st.download_button(
            "💾 Bu tahmini indir",
            result.to_csv(index=False).encode("utf-8"),
            file_name="tek_kayit_tahmin.csv",
            mime="text/csv"
        )
