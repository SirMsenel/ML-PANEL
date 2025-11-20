import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ML Analiz Paneli",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Makine Öğrenmesi Analiz Paneli")
st.markdown("""
Bu panel, kullanıcı dostu bir arayüzle veri analizi ve makine öğrenmesi modellerini 
tek bir platform üzerinden çalıştırmanızı sağlar.
""")

st.image("https://cdn-icons-png.flaticon.com/512/1087/1087840.png", width=120)

st.markdown("### 🚀 Başlıca Özellikler")
col1, col2, col3 = st.columns(3)

with col1:
    st.success("📂 Veri Yükleme ve Ön İzleme")
with col2:
    st.info("📊 Temel Analizler ve Görselleştirme")
with col3:
    st.warning("🤖 Makine Öğrenmesi Modelleri")

st.markdown("---")
st.write("👈 Sol menüden ilerleyerek veri yükleyebilir ve analiz adımlarına geçebilirsiniz.")
st.caption("Geliştiren: Mehmet Şenel | @2025")



 