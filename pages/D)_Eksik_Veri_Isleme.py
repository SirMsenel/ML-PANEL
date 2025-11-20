# pages/eksik_veri_isleme.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.title("🧩 Eksik Veri İşleme")
st.markdown("---")

# --- Veri kontrolü ---
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"].copy()
else:
    st.warning("⚠️ Önce veri yükleyin! '📂 Veri Yükleme' sayfasına gidin.")
    st.stop()

# --- Eksik veri özeti ---
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]

# Eğer eksik veri yoksa bile "İşlemi Geri Al" ve "Veriyi İndir" görünsün
if missing_summary.empty:
    st.success("✅ Veri setinde eksik değer bulunmamaktadır. İşleme gerek yok.")
    if "backup_df" in st.session_state:
        if st.button("↩️ İşlemi Geri Al"):
            st.session_state["uploaded_df"] = st.session_state["backup_df"]
            st.session_state["degisen_satirlar"] = pd.DataFrame()
            st.success("✅ İşlem geri alındı, veri seti önceki haline döndü.")
            st.rerun()

    # 💾 Güncellenmiş veriyi indir
    buffer = BytesIO()
    st.session_state["uploaded_df"].to_csv(buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="💾 Güncellenmiş Veriyi İndir",
        data=buffer.getvalue(),
        file_name="guncellenmis_veri.csv",
        mime="text/csv"
    )
    st.stop()

summary_df = pd.DataFrame({
    "Sütun": missing_summary.index,
    "Eksik Sayısı": missing_summary.values,
    "Oran (%)": np.round((missing_summary.values / len(df)) * 100, 2),
    "Veri Tipi": [df[col].dtype for col in missing_summary.index]
})

# 🔹 Eksik veri tablosu (grafikten önce)
st.subheader("📊 Eksik Veri Özeti")
st.dataframe(summary_df, use_container_width=True)
st.markdown("---")

# 🔹 Eksik veri içeren satırlar (grafikten önce)
st.subheader("🔍 Eksik Veri İçeren Satırlar")
missing_rows = df[df.isnull().any(axis=1)]
if not missing_rows.empty:
    def highlight_missing(val):
        if pd.isnull(val):
            return 'background-color: #ff8080; color: black;'
        return ''
    st.dataframe(missing_rows.style.applymap(highlight_missing), use_container_width=True, height=400)
    st.caption(f"🧩 Toplam {len(missing_rows)} satırda eksik veri bulunuyor.")
else:
    st.success("✅ Veri setinde eksik değer bulunmamaktadır.")

st.markdown("---")

# 🔹 Eksik veri görselleştirme
st.subheader("⚠️ Eksik Veri Durumu")
missing_per_column = df.isnull().sum()
missing_per_column = missing_per_column[missing_per_column > 0]

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Sütun Bazlı Eksik Veri")
    if not missing_per_column.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=missing_per_column.index, y=missing_per_column.values, palette="Reds_r", ax=ax)
        ax.set_ylabel("Eksik Hücre Sayısı")
        ax.set_xlabel("Sütunlar")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        for i, v in enumerate(missing_per_column.values):
            ax.text(i, v + 0.5, str(v), ha='center', color='black', fontweight='bold')
        st.pyplot(fig)
    else:
        st.info("✅ Tüm sütunlarda eksik veri yok.")

with col2:
    st.markdown("### 🥧 Genel Eksik / Dolu Hücre Oranı")
    total_cells = df.shape[0] * df.shape[1]
    missing_count = df.isnull().sum().sum()
    filled_count = total_cells - missing_count

    if missing_count == 0:
        st.info("✅ Veri setinde eksik hücre yok.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 5.1))
        ax2.pie(
            [filled_count, missing_count],
            labels=["Dolu Hücre", "Eksik Hücre"],
            autopct=lambda p: f"{p:.1f}%\n({int(p * total_cells / 100)})",
            colors=["#8BC34A", "#F44336"],
            startangle=90,
            textprops={'color': "black", 'fontsize': 12}
        )
        ax2.axis('equal')
        st.pyplot(fig2)

st.markdown("---")

# --- İşlem türü seçimi ---
st.subheader("⚙️ İşlem Türü Seçimi")
islem = st.radio("Ne yapmak istersiniz?", ["Eksik Veriyi Doldur", "Eksik Veriyi Sil"], horizontal=True)
st.markdown("---")

# ---------------------- #
# --- DOLDURMA BLOĞU --- #
# ---------------------- #
if islem == "Eksik Veriyi Doldur":
    st.subheader("🧠 Eksik Veriyi Doldurma")
    col1, col2 = st.columns(2)
    st.info("Gerekmedikçe 'Tümü' seçeneğini kullanmayın")

    with col1:
        secilen_sutun = st.selectbox("Sütun Seçin", ["Tümü"] + list(missing_summary.index))

    if secilen_sutun == "Tümü":
        yontem_options = ["Mod (mode)", "Sabit Değer Gir", "Ortalama (mean)", "Medyan (median)"]
    else:
        dtype = df[secilen_sutun].dtype
        if np.issubdtype(dtype, np.number):
            yontem_options = ["Ortalama (mean)", "Medyan (median)", "Mod (mode)", "Sabit Değer Gir"]
        else:
            yontem_options = ["Mod (mode)", "Sabit Değer Gir"]

    with col2:
        yontem = st.selectbox("Doldurma Yöntemi", yontem_options)

    sabit_deger = None
    if yontem == "Sabit Değer Gir":
        sabit_deger = st.text_input("Sabit değeri girin")

    if st.button("🚀 Doldurmayı Uygula"):
        df_filled = df.copy()
        target_cols = missing_summary.index if secilen_sutun == "Tümü" else [secilen_sutun]
        doldurulan = 0
        degisen_satirlar = pd.DataFrame()

        for col in target_cols:
            if df[col].isnull().sum() == 0:
                continue

            is_numeric = np.issubdtype(df[col].dtype, np.number)

            if yontem == "Ortalama (mean)" and is_numeric:
                missing_idx = df[col][df[col].isnull()].index
                df_filled.loc[missing_idx, col] = df[col].mean()
                doldurulan += len(missing_idx)
                degisen_satirlar = pd.concat([degisen_satirlar, df_filled.loc[missing_idx]])

            elif yontem == "Medyan (median)" and is_numeric:
                missing_idx = df[col][df[col].isnull()].index
                df_filled.loc[missing_idx, col] = df[col].median()
                doldurulan += len(missing_idx)
                degisen_satirlar = pd.concat([degisen_satirlar, df_filled.loc[missing_idx]])

            elif yontem == "Mod (mode)":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    missing_idx = df[col][df[col].isnull()].index
                    df_filled.loc[missing_idx, col] = mode_val[0]
                    doldurulan += len(missing_idx)
                    degisen_satirlar = pd.concat([degisen_satirlar, df_filled.loc[missing_idx]])

            elif yontem == "Sabit Değer Gir" and sabit_deger != "":
                missing_idx = df[col][df[col].isnull()].index
                df_filled.loc[missing_idx, col] = sabit_deger
                doldurulan += len(missing_idx)
                degisen_satirlar = pd.concat([degisen_satirlar, df_filled.loc[missing_idx]])

        # Geçmişi sakla
        st.session_state["backup_df"] = df.copy()
        st.session_state["uploaded_df"] = df_filled
        st.session_state["degisen_satirlar"] = degisen_satirlar
        st.success(f"✅ Doldurma işlemi tamamlandı. {doldurulan} hücre dolduruldu.")
        st.rerun()

# -------------------- #
# --- SİLME BLOĞU --- #
# -------------------- #
else:
    st.subheader("🗑️ Eksik Veriyi Silme")
    silme_turu = st.radio("Silme yöntemi seçin:", [
        "Eksik değer içeren satırları sil",
        "Eksik değer içeren sütunları sil",
        "Seçili sütundaki eksik satırları sil"
    ])

    secilen_sutun = None
    if silme_turu == "Seçili sütundaki eksik satırları sil":
        secilen_sutun = st.selectbox("Sütun seçin", list(missing_summary.index))

    if st.button("🚀 Silme İşlemini Uygula"):
        df_sil = df.copy()
        degisen_satirlar = pd.DataFrame()

        if silme_turu == "Eksik değer içeren satırları sil":
            missing_idx = df_sil[df_sil.isnull().any(axis=1)].index
            degisen_satirlar = df_sil.loc[missing_idx]
            df_sil.dropna(inplace=True)

        elif silme_turu == "Eksik değer içeren sütunları sil":
            degisen_satirlar = df_sil[df_sil.columns[df_sil.isnull().any()]]
            df_sil.dropna(axis=1, inplace=True)

        elif secilen_sutun:
            missing_idx = df_sil[df_sil[secilen_sutun].isnull()].index
            degisen_satirlar = df_sil.loc[missing_idx]
            df_sil = df_sil[df_sil[secilen_sutun].notnull()]

        st.session_state["backup_df"] = df.copy()
        st.session_state["uploaded_df"] = df_sil
        st.session_state["degisen_satirlar"] = degisen_satirlar
        st.success("✅ Silme işlemi tamamlandı.")
        st.rerun()

st.markdown("---")

# 🔹 Değişiklik yapılan satırları göster
if "degisen_satirlar" in st.session_state and not st.session_state["degisen_satirlar"].empty:
    st.subheader("🔍 Değişiklik Yapılan Satırlar")
    st.dataframe(
        st.session_state["degisen_satirlar"].style.applymap(
            lambda v: 'background-color: #ff8080; color: black;' if pd.isnull(v) else ''
        ),
        use_container_width=True,
        height=300
    )

# 🔹 Geri al butonu
if "backup_df" in st.session_state:
    if st.button("↩️ İşlemi Geri Al"):
        st.session_state["uploaded_df"] = st.session_state["backup_df"]
        st.session_state["degisen_satirlar"] = pd.DataFrame()
        st.success("✅ İşlem geri alındı, veri seti önceki haline döndü.")
        st.rerun()

# 💾 Güncellenmiş veriyi indir
buffer = BytesIO()
st.session_state["uploaded_df"].to_csv(buffer, index=False, encoding="utf-8-sig")
st.download_button(
    label="💾 Güncellenmiş Veriyi İndir",
    data=buffer.getvalue(),
    file_name="guncellenmis_veri.csv",
    mime="text/csv"
)
