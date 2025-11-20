import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




st.set_page_config(page_title="Veri Yükleme", page_icon="📁", layout="wide")

st.title("📊 Veri Yükleme ve Ön İnceleme")
st.markdown("Bu sayfada verinizi yükleyebilir, temel yapı ve istatistikleri görüntüleyebilirsiniz.")
st.markdown("---")

# 🔹 Dosya yükleme alanı
uploaded_file = st.file_uploader(
    "Bir veri dosyası yükleyin (.csv veya .xlsx)",
    type=["csv", "xlsx"],
    key="uploaded_file"
)

# Eğer yeni dosya yüklenmişse session_state'i güncelle
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Her zaman session_state güncelleniyor
        st.session_state["uploaded_df"] = df
        st.success(f"✅ Veri '{uploaded_file.name}' başarıyla yüklendi!")

    except Exception as e:
        st.error(f"❌ Dosya yüklenirken hata oluştu: {e}")
        df = None

# Eğer dosya yüklenmemişse session_state’deki veriyi kullan
elif "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
else:
    df = None
    st.info("Lütfen bir CSV veya Excel dosyası yükleyin.")


# Eğer session_state’de veri varsa bunu kullan
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
elif uploaded_file is not None:
    try:
        # 🔸 Dosya tipine göre okuma
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)

        # Otomatik session_state kaydı
        st.session_state["uploaded_df"] = df
        st.success("✅ Veri başarıyla yüklendi ve session_state'e kaydedildi!")

    except Exception as e:
        st.error(f"❌ Dosya yüklenirken hata oluştu: {e}")
        df = None


# Eğer veri yüklüyse devam et
if df is not None:

    # 🔹 Temel Bilgiler - metric ile vurgulu
    st.subheader("📋 Temel Bilgiler")
    total_cells = df.shape[0] * df.shape[1]
    missing_count = df.isnull().sum().sum()
    missing_percent = round((missing_count / total_cells) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div style="background-color:#E5F8E0; padding:10px; border-radius:8px; text-align:center">
    <h4 style="color:black;">🧩 Toplam Veri</h4>
    <p style="font-size:24px; font-weight:bold; color:black;">{total_cells:,}</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="background-color:#E8F0FE; padding:10px; border-radius:8px; text-align:center">
    <h4 style="color:black;">📄 Satır Sayısı</h4>
    <p style="font-size:24px; font-weight:bold; color:black;">{df.shape[0]:,}</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="background-color:#FFF4E5; padding:10px; border-radius:8px; text-align:center">
    <h4 style="color:black;">📊 Sütun Sayısı</h4>
    <p style="font-size:24px; font-weight:bold; color:black;">{df.shape[1]:,}</p>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style="background-color:#FEE5E5; padding:10px; border-radius:8px; text-align:center">
    <h4 style="color:black;">⚠️ Eksik Hücre (%)</h4>
    <p style="font-size:24px; font-weight:bold; color:black;">{missing_percent}%</p>
    </div>
    """, unsafe_allow_html=True)


    # 🔹 Veri önizlemesi
    st.subheader("🔍 Veri Önizleme")
    row_count = st.slider("Kaç satır görmek istersiniz?", 5, len(df), 5)
    st.dataframe(df.head(row_count))


    # 🔹 Sütun bazlı bilgiler
    st.subheader("🧩 Sütun Bilgileri")
    column_info = pd.DataFrame({
        "Toplam Değer": len(df),
        "Dolu Hücre Sayısı": df.notnull().sum(),
        "Eksik Hücre Sayısı": df.isnull().sum(),
        "Eksik Oran (%)": round(df.isnull().mean() * 100 , 2)
    }).reset_index().rename(columns={"index": "Sütun Adı"})

    # --- Stil fonksiyonu ---
    def highlight_missing(val):
        if val > 0:
            return 'background-color: #FFCDD2; color: black; font-weight: bold'
        else:
            return 'background-color: #C8E6C9; color: black; font-weight: bold'

    styled_info = column_info.style.applymap(highlight_missing, subset=["Eksik Hücre Sayısı"])
    st.dataframe(styled_info, use_container_width=True)




    # 🔹 Veri tipleri ve kategorik / sayısal belirleme
    st.subheader("🔢 Sütun Tipleri ve Benzersiz Değer Sayısı")
    column_summary = pd.DataFrame({
        "Toplam Değer": len(df),
        "Benzersiz Değer": df.nunique(),
        "Veri Tipi": df.dtypes.astype(str),
    }).reset_index().rename(columns={"index": "Sütun Adı"})

    def categorize_dtype(dtype, nunique):
        if pd.api.types.is_numeric_dtype(dtype):
            return "Sayısal"
        elif pd.api.types.is_string_dtype(dtype) or nunique < 20:
            return "Kategorik"
        else:
            return "Diğer"

    column_summary["Tür"] = column_summary.apply(
        lambda x: categorize_dtype(df[x["Sütun Adı"]].dtype, x["Benzersiz Değer"]), axis=1
    )

    st.dataframe(column_summary, use_container_width=True)
