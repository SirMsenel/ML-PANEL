# pages/aykiri_deger_isleme.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("⚡ Aykırı Değer Tespiti ve İşleme")
st.markdown("---")

# --- Veri kontrolü ---
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"].copy()
else:
    st.warning("⚠️ Önce veri yükleyin! '📂 Veri Yükleme' sayfasına gidin.")
    st.stop()

# --- Sayısal sütunlar ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if not numeric_cols:
    st.info("✅ Veri setinde sayısal sütun bulunmamaktadır.")
    st.stop()

st.subheader("🔹 Parametreler")

col1, col2 = st.columns(2)
with col1:
    secilen_sutun = st.selectbox("Aykırı değer tespiti için sütun seçin", numeric_cols)
with col2:
    grup_sutunlar = st.multiselect("Opsiyonel: Gruplama sütun(ları) seçin", df.columns.tolist())

iqr_factor = st.slider("IQR Çarpanı (Alt/Üst sınır için)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

st.markdown("---")

# --- Fonksiyonlar ---
def detect_outliers(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outliers = (series < lower) | (series > upper)
    return outliers, lower, upper

def update_outliers(df, secilen_sutun, grup_sutunlar, iqr_factor):
    alt_ust_list = []
    aykiri_df_list = []

    if grup_sutunlar:
        grouped = df.groupby(grup_sutunlar)
        for name, group in grouped:
            outliers, lower, upper = detect_outliers(group[secilen_sutun], factor=iqr_factor)
            temp = group.copy()
            aykiri_df_list.append(temp[outliers])
            group_name = ", ".join(map(str, name)) if isinstance(name, tuple) else str(name)
            alt_ust_list.append([group_name, lower, upper])
    else:
        outliers, lower, upper = detect_outliers(df[secilen_sutun], factor=iqr_factor)
        aykiri_df_list.append(df[outliers].copy())
        alt_ust_list.append(["Tümü", lower, upper])

    aykiri_df = pd.concat(aykiri_df_list) if aykiri_df_list else pd.DataFrame()
    alt_ust_df = pd.DataFrame(alt_ust_list, columns=["Grup", "Alt Sınır", "Üst Sınır"])
    return aykiri_df, alt_ust_df

# --- Güncel tablo oluştur ---
aykiri_df, alt_ust_df = update_outliers(df, secilen_sutun, grup_sutunlar, iqr_factor)

# --- Alt/Üst sınır tablosu ---
st.subheader("📋 Alt/Üst Sınır Tablosu")
st.dataframe(alt_ust_df, use_container_width=True)
st.markdown("---")

# --- Aykırı değer tablosu ---
st.subheader("⚠️ Aykırı Değerler")
if aykiri_df.empty:
    st.success("✅ Seçilen sütunda aykırı değer bulunmamaktadır.")
else:
    st.dataframe(aykiri_df, use_container_width=True, height=400)
    st.info(f"Toplam {len(aykiri_df)} aykırı değer bulundu.")

st.markdown("---")
st.subheader("🛠️ Aykırı Değer İşlemleri")

# --- İşlem Seçenekleri ---
st.markdown("Seçilen aykırı değerler için işlem seçin (tek seçim yapabilirsiniz):")
aykiri_islem = st.radio(
    "",
    [
        "Alt/Üst sınıra eşitle",
        "Satırı Sil",
        "Ortalama ile doldur",
        "Medyan ile doldur",
        "Mod ile doldur",
        "Aykırı değer sütunu ekle (0/1 işaretleme)"
    ]
)

# --- İşlem butonu ---
if st.button("🚀 İşlemi Uygula"):
    df_updated = st.session_state["uploaded_df"].copy()
    idx = aykiri_df[secilen_sutun].index
    degisen_satirlar = aykiri_df.copy()

    for i in idx:
        if i not in df_updated.index:
            continue

        val = df_updated.loc[i, secilen_sutun]

        if grup_sutunlar:
            # Gruplama anahtarına göre alt/üst sınır bul
            row_group = tuple(df_updated.loc[i, grup_sutunlar]) if len(grup_sutunlar) > 1 else df_updated.loc[i, grup_sutunlar[0]]
            group_name = ", ".join(map(str, row_group)) if isinstance(row_group, tuple) else str(row_group)
            match = alt_ust_df[alt_ust_df["Grup"] == group_name]
            alt, ust = match["Alt Sınır"].values[0], match["Üst Sınır"].values[0]
        else:
            alt, ust = alt_ust_df.loc[0, ["Alt Sınır", "Üst Sınır"]]

        if aykiri_islem == "Alt/Üst sınıra eşitle":
            if val < alt:
                df_updated.loc[i, secilen_sutun] = alt
            elif val > ust:
                df_updated.loc[i, secilen_sutun] = ust
        elif aykiri_islem == "Satırı Sil":
            df_updated.drop(index=i, inplace=True)
        elif aykiri_islem == "Ortalama ile doldur":
            df_updated.loc[i, secilen_sutun] = df[secilen_sutun].mean()
        elif aykiri_islem == "Medyan ile doldur":
            df_updated.loc[i, secilen_sutun] = df[secilen_sutun].median()
        elif aykiri_islem == "Mod ile doldur":
            df_updated.loc[i, secilen_sutun] = df[secilen_sutun].mode()[0]
        elif aykiri_islem == "Aykırı değer sütunu ekle (0/1 işaretleme)":
            col_name = "Aykırı_" + secilen_sutun
            if col_name not in df_updated.columns:
                df_updated[col_name] = 0
            df_updated.loc[idx, col_name] = 1

    # --- Session state güncelle ---
    st.session_state["uploaded_df"] = df_updated
    st.session_state["backup_df"] = df.copy()
    st.session_state["degisen_satirlar_aykiri"] = degisen_satirlar

    st.rerun()

# --- Değişiklik tablosu (eski ve yeni değerlerle birlikte) ---
if "degisen_satirlar_aykiri" in st.session_state and not st.session_state["degisen_satirlar_aykiri"].empty:
    st.subheader("🔍 Değişiklik Yapılan Satırlar (Eski ve Yeni Değerler)")

    eski = st.session_state["degisen_satirlar_aykiri"]
    yeni = st.session_state["uploaded_df"].loc[eski.index, eski.columns]

    eski_rename = eski.add_prefix("Eski_")
    yeni_rename = yeni.add_prefix("Yeni_")

    degisen_full = pd.concat([eski_rename, yeni_rename], axis=1)
    st.dataframe(degisen_full, use_container_width=True, height=300)

# --- Geri al butonu ---
if "backup_df" in st.session_state and st.button("↩️ İşlemi Geri Al"):
    st.session_state["uploaded_df"] = st.session_state["backup_df"]
    st.session_state["degisen_satirlar_aykiri"] = pd.DataFrame()
    st.success("✅ İşlem geri alındı, veri seti önceki haline döndü.")
    st.rerun()

# --- Veri dışa aktarma ---
st.markdown("---")
st.subheader("💾 Güncellenmiş Veriyi İndir")
csv = st.session_state["uploaded_df"].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 CSV olarak indir",
    data=csv,
    file_name="guncellenmis_veri.csv",
    mime="text/csv"
)
