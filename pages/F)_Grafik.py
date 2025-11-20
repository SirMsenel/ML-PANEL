import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



st.set_page_config(page_title="Grafikler", page_icon="🎨", layout="wide")
st.title("🎨 Grafikler")

# Session state'den veri çek
if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"]
else:
    st.warning("Önce veri yükleyin!")
    st.stop()

# Sayısal ve kategorik sütunları ayır
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()


# ------------------- GÖRSELLEŞTİRME -------------------

# Grafik seçenekleri
numeric_graphs = ["Histogram", "Boxplot", "Violin Plot", "Density Plot", "Scatter","Trendline"]
categorical_graphs = ["Bar Grafiği", "Pasta Grafiği"]
num_cat_graphs = ["Boxplot by Category", "Violin by Category", "Scatter by Category"]

col_type = st.radio("Hangi tip sütunları görselleştirmek istersiniz?", 
                    ("Sayısal", "Kategorik", "Sayısal + Kategorik"))

# ------------------- SAYISAL -------------------
if col_type == "Sayısal":
    if not numeric_cols:
        st.warning("⚠️ Sayısal sütun bulunamadı.")
    else:
        selected_graph = st.selectbox("Grafik tipi", numeric_graphs)
        x_col = st.selectbox("X ekseni seçin", numeric_cols, index=0)
        y_col = None

        if selected_graph in ["Scatter", "Trendline"]:
            y_col = st.selectbox("Y ekseni seçin", numeric_cols, index=0)

        fig, ax = plt.subplots(figsize=(12,5))

        if selected_graph == "Histogram":
            sns.histplot(df[x_col].dropna(), kde=True, ax=ax, color="#4CAF50")
        elif selected_graph == "Boxplot":
            sns.boxplot(x=df[x_col], ax=ax, color="#FFC107")
        elif selected_graph == "Violin Plot":
            sns.violinplot(x=df[x_col], ax=ax, color="#9C27B0")
        elif selected_graph == "Density Plot":
            sns.kdeplot(df[x_col].dropna(), ax=ax, fill=True, color="#03A9F4")
        elif selected_graph == "Scatter":
                sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color="#9C27B0")
        elif selected_graph == "Trendline":
                if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                    sns.lmplot(x=x_col, y=y_col, data=df, aspect=2, height=5)
                    st.pyplot(plt.gcf())
                    plt.close()
                    st.stop()  # lmplot ayrı figür olduğundan normal figürü çizme
                else:
                    st.error("❌ Trendline için X ve Y sayısal olmalı.")
        st.pyplot(fig)

# ------------------- KATEGORİK -------------------
elif col_type == "Kategorik":
    if not categorical_cols:
        st.warning("⚠️ Kategorik sütun bulunamadı.")
    else:
        selected_graph = st.selectbox("Grafik tipi", categorical_graphs)
        x_col = st.selectbox("X ekseni seçin", categorical_cols, index=0)

        fig, ax = plt.subplots(figsize=(10,5))
        if selected_graph == "Bar Grafiği":
            sns.countplot(y=df[x_col], order=df[x_col].value_counts().index, palette="Set2", ax=ax)
            ax.set_xlabel("Frekans")
            ax.set_ylabel(x_col)
        elif selected_graph == "Pasta Grafiği":
            counts = df[x_col].value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, textprops={'color':"black"})
            ax.axis('equal')
        st.pyplot(fig)

# ------------------- SAYISAL + KATEGORİK -------------------
elif col_type == "Sayısal + Kategorik":
    if not numeric_cols or not categorical_cols:
        st.warning("⚠️ Hem sayısal hem kategorik sütun bulunmalı.")
    else:
        selected_graph = st.selectbox("Grafik tipi", num_cat_graphs)
        x_col = st.selectbox("X ekseni (Kategorik)", categorical_cols, index=0)
        y_col = st.selectbox("Y ekseni (Sayısal)", numeric_cols, index=0)

        fig, ax = plt.subplots(figsize=(10,5))
        if selected_graph == "Boxplot by Category":
            sns.boxplot(x=df[x_col], y=df[y_col], ax=ax, palette="Set3")
        elif selected_graph == "Violin by Category":
            sns.violinplot(x=df[x_col], y=df[y_col], ax=ax, palette="Set2")
        elif selected_graph == "Scatter by Category":
            sns.scatterplot(x=df[x_col].astype(str), y=df[y_col], hue=df[x_col].astype(str), ax=ax, palette="tab10")
        st.pyplot(fig)