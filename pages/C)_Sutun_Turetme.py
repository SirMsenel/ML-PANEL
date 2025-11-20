import streamlit as st
import pandas as pd
import numpy as np

st.title("🧱 Yeni Sütun / Türetme ve Silme İşlemleri")
st.markdown("---")

# --- Veri kontrolü ---
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"].copy()
else:
    st.warning("⚠️ Önce veri yükleyin!")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
date_cols = df.select_dtypes(include=["datetime64[ns]", "object"]).columns.tolist()

# --- İşlem türü seçimi ---
islem = st.selectbox("İşlem türü seçin", [
    "Sabit Değer ile Yeni Sütun",
    "İki Sayısal Sütunu Birleştir",
    "Sütunu Kategoriye Dönüştür (Binning)",
    "Tarih Sütunundan Yeni Öznitelikler",
    "Eksik Değer Bayrak Sütunu (Null Flag)",
    "Sütun Sil"
])

st.markdown("---")

# --- 1️⃣ Sabit değer ---
if islem == "Sabit Değer ile Yeni Sütun":
    col_ad = st.text_input("Yeni sütun adı:")
    sabit = st.text_input("Sabit değer:")
    if st.button("➕ Oluştur"):
        df[col_ad] = sabit
        st.session_state["uploaded_df"] = df
        st.success(f"✅ '{col_ad}' sütunu başarıyla eklendi!")

# --- 2️⃣ Sayısal sütun birleştirme ---
elif islem == "İki Sayısal Sütunu Birleştir":
    s1 = st.selectbox("1. sütun", numeric_cols)
    s2 = st.selectbox("2. sütun", numeric_cols)
    op = st.selectbox("İşlem", ["Topla", "Çıkar", "Çarp", "Böl"])
    yeni = st.text_input("Yeni sütun adı:")
    if st.button("➕ Oluştur"):
        if op == "Topla": df[yeni] = df[s1] + df[s2]
        elif op == "Çıkar": df[yeni] = df[s1] - df[s2]
        elif op == "Çarp": df[yeni] = df[s1] * df[s2]
        elif op == "Böl": df[yeni] = df[s1] / df[s2].replace(0, np.nan)
        st.session_state["uploaded_df"] = df
        st.success(f"✅ '{yeni}' sütunu başarıyla oluşturuldu!")

# --- 3️⃣ Sayıları kategoriye dönüştürme (binning) ---
elif islem == "Sütunu Kategoriye Dönüştür (Binning)":
    col = st.selectbox("Sütun seç", numeric_cols)
    bins = st.slider("Kategori sayısı", 2, 10, 4)
    yeni = st.text_input("Yeni sütun adı:", col + "_kategori")

    if st.button("➕ Oluştur"):
        # Binning işlemi (Interval değil, temiz sayısal etiketler)
        df[yeni] = pd.qcut(df[col], bins, labels=range(1, bins + 1), duplicates='drop')

        # Etiketleri INT yapmak için:
        df[yeni] = df[yeni].astype(int)

        st.session_state["uploaded_df"] = df
        st.success(f"✅ '{yeni}' kategorik sütunu başarıyla oluşturuldu! (Etiketler: 1-{bins})")

        
# --- 4️⃣ Tarih sütunu -> genişletilmiş öznitelikler ---
elif islem == "Tarih Sütunundan Yeni Öznitelikler":
    date_col = st.selectbox("Tarih sütunu seç", date_cols)
    yeni_tur = st.multiselect(
        "Oluşturulacak özellikler",
        [
            "Yıl", "Ay", "Gün", "Hafta", "Çeyrek",
            "Haftanın Günü", "Hafta İçi/Hafta Sonu", "Ayın Haftası",  # ✅ YENİ
            "Ay Adı", "Yıl-Ay (Period)",
            "Saat", "Dakika", "Saniye"
        ]
    )

    if st.button("➕ Oluştur"):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        if "Yıl" in yeni_tur: df[f"{date_col}_yil"] = df[date_col].dt.year
        if "Ay" in yeni_tur: df[f"{date_col}_ay"] = df[date_col].dt.month
        if "Gün" in yeni_tur: df[f"{date_col}_gun"] = df[date_col].dt.day
        if "Hafta" in yeni_tur: df[f"{date_col}_hafta"] = df[date_col].dt.isocalendar().week.astype(int)
        if "Çeyrek" in yeni_tur: df[f"{date_col}_ceyrek"] = df[date_col].dt.quarter
        if "Haftanın Günü" in yeni_tur: df[f"{date_col}_haftanin_gunu"] = df[date_col].dt.day_name()

        # ✅ Hafta İçi (1) / Hafta Sonu (0)
        if "Hafta İçi/Hafta Sonu" in yeni_tur:
            df[f"{date_col}_haftaici"] = df[date_col].dt.weekday < 5  # True=Hafta içi
            df[f"{date_col}_haftaici"] = df[f"{date_col}_haftaici"].astype(int)

        # ✅ Ayın Kaçıncı Haftası
        if "Ayın Haftası" in yeni_tur:
            df[f"{date_col}_ayin_haftasi"] = ((df[date_col].dt.day - 1) // 7) + 1

        if "Ay Adı" in yeni_tur: df[f"{date_col}_ay_adi"] = df[date_col].dt.month_name()
        if "Yıl-Ay (Period)" in yeni_tur:
            df[f"{date_col}_yil_ay"] = df[date_col].dt.to_period("M").astype(str)

        # ✅ SAAT BİLGİSİ
        if "Saat" in yeni_tur: df[f"{date_col}_saat"] = df[date_col].dt.hour
        if "Dakika" in yeni_tur: df[f"{date_col}_dakika"] = df[date_col].dt.minute
        if "Saniye" in yeni_tur: df[f"{date_col}_saniye"] = df[date_col].dt.second

        st.session_state["uploaded_df"] = df
        st.success("✅ Tarih & Saat özellikleri başarıyla oluşturuldu!")

# --- 5️⃣ Null Flag Eklenmesi ---
elif islem == "Eksik Değer Bayrak Sütunu (Null Flag)":
    col = st.selectbox("Sütun seç", df.columns)
    yeni = col + "_is_null"
    if st.button("➕ Oluştur"):
        df[yeni] = df[col].isnull().astype(int)
        st.session_state["uploaded_df"] = df
        st.success(f"✅ '{yeni}' sütunu oluşturuldu! (0 = Dolu, 1 = Eksik)")

# --- 6️⃣ Sütun Silme ---
elif islem == "Sütun Sil":
    silinecek = st.multiselect("Silinecek sütun(lar)ı seçin", df.columns)
    if st.button("🗑️ Sütunları Sil"):
        df.drop(columns=silinecek, inplace=True)
        st.session_state["uploaded_df"] = df
        st.success(f"🧹 {', '.join(silinecek)} sütun(lar)ı silindi.")
