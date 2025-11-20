import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st

# ------------------------------------------------
# Sayfa ayarları
# ------------------------------------------------
st.set_page_config(page_title="Temel Analizler", page_icon="📊", layout="wide")
st.title("📊 Genişletilmiş Temel Analizler ve Görselleştirme")
st.markdown("---")

# ------------------------------------------------
# Veri kontrolü
# ------------------------------------------------
if "uploaded_df" not in st.session_state:
    st.warning("⚠️ Önce veri yükleyin!")
    st.stop()

df = st.session_state["uploaded_df"].copy()

# Sayısal / Kategorik sütun ayrımı
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# ==============================================================
# 📈 SAYISAL ANALİZLER
# ==============================================================
st.subheader("📈 Sayısal Sütun Analizi")
st.markdown("---")

if numeric_cols:
    # 🔹 Varsayılan: hiçbir sütun seçili değil
    selected_numeric = st.multiselect("Sayısal sütun(lar) seçin", numeric_cols, default=[])
    group_col = st.multiselect("Opsiyonel: Bir veya birden fazla sütunla grupla", categorical_cols)


    if selected_numeric:
        if group_col:
            st.markdown(f"**📊 {group_col} bazında özet istatistikler:**")
            grouped_stats = df.groupby(group_col)[selected_numeric].describe().T
            st.dataframe(grouped_stats, use_container_width=True)
        else:
            st.markdown("**📊 Genel özet istatistikler:**")
            st.dataframe(df[selected_numeric].describe().T, use_container_width=True)

        st.markdown("---")

        # Hareketli Ortalama Hesaplama
        st.subheader("📉 Hareketli Ortalama Analizi (Opsiyonel)")
        time_col = st.selectbox("Zaman bazlı analiz için tarih veya sıra sütunu seçin", [None] + df.columns.tolist())
        window_size = st.slider("Hareketli ortalama pencere boyutu", 2, 30, 5)

        if time_col and st.button("Hareketli Ortalama Grafiğini Göster"):
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in selected_numeric:
                if np.issubdtype(df[time_col].dtype, np.number):
                    x_vals = df[time_col]
                else:
                    x_vals = pd.to_datetime(df[time_col], errors="coerce")

                ax.plot(x_vals, df[col], alpha=0.4, label=f"{col} (Orijinal)")
                ax.plot(x_vals, df[col].rolling(window=window_size).mean(), label=f"{col} (MA-{window_size})", linewidth=2.5)
            ax.set_title("Hareketli Ortalama Grafiği")
            ax.legend()
            st.pyplot(fig)
else:
    st.info("⚠️ Sayısal sütun bulunamadı.")
st.markdown("---")

# ==============================================================
# 🧩 KATEGORİK ANALİZLER
# ==============================================================
# ------------------- KATEGORİK BLOĞU (GÜNCELLEME) -------------------
st.subheader("🧩 Kategorik Sütun Analizi")
st.markdown("---")

if categorical_cols:
    selected_categorical = st.multiselect(
        "Kategorik sütun(lar) seçin", 
        categorical_cols, 
        default=[]
    )

    group_for_cat = st.multiselect(
        "Opsiyonel: Kategorileri grupla (birden fazla seçilebilir)", 
        categorical_cols
    )

    if selected_categorical:
        for col in selected_categorical:
            st.markdown(f"### 🔸 {col}")

            # Frekans tablosu (temel)
            freq = df[col].value_counts(dropna=False)
            perc = (freq / len(df) * 100).round(2)
            freq_table = pd.DataFrame({"Frekans": freq, "Oran (%)": perc})

            # Eğer gruplama seçiliyse -> pd.crosstab ile güvenli pivot
            if group_for_cat:
                safe_groups = [g for g in group_for_cat if g != col]

                if not safe_groups:
                    st.warning("Gruplama sütunları içinde analiz edilen sütun seçilmiş; lütfen farklı grup sütunları seçin.")
                    st.dataframe(freq_table.head(10), use_container_width=True)
                    st.bar_chart(freq)
                else:
                    try:
                        # Crosstab oluştur
                        index_list = [df[g] for g in safe_groups]
                        pivot_df = pd.crosstab(index=index_list, columns=df[col], dropna=False)

                        # NaN değerleri güvenli şekilde doldur
                        pivot_df = pivot_df.fillna(0)

                        # Index ve kolon isimlerindeki NaN'leri düzelt
                        pivot_df.index = pivot_df.index.to_frame(index=False).fillna("Eksik").astype(str).agg(" | ".join, axis=1)
                        pivot_df.columns = pivot_df.columns.to_series().fillna("Eksik").astype(str)

                        # Sade bir index oluştur
                        pivot_df = pivot_df.reset_index(names="Grup")

                        st.markdown("**📋 Gruplamalı Frekans (crosstab)**")
                        st.dataframe(pivot_df, use_container_width=True)

                        # Yüzde gösterimi (opsiyonel)
                        show_pct = st.checkbox(f"% göster: {col} (gruplara göre)", key=f"pct_{col}")
                        if show_pct:
                            pct_df = pivot_df.set_index("Grup")
                            pct_df = pct_df.div(pct_df.sum(axis=1), axis=0).multiply(100).round(2).reset_index()
                            st.markdown("**% Dağılım (Satır bazlı)**")
                            st.dataframe(pct_df, use_container_width=True)

                        # ---------------- Grafik kısmı ----------------
                        st.markdown("**📊 Gruplamalı Grafik (stacked)**")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        pivot_df_plot = pivot_df.set_index("Grup")
                        pivot_df_plot.plot(kind="bar", stacked=True, ax=ax)
                        ax.set_ylabel("Frekans")
                        ax.set_title(f"{', '.join(safe_groups)} bazında {col} dağılımı")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Pivot tablo oluşturulurken hata oluştu: {e}")
                        st.dataframe(freq_table.head(10), use_container_width=True)
            else:
                # Gruplama yoksa sade tablo + grafik
                st.dataframe(freq_table, use_container_width=True)
                st.bar_chart(freq)

            st.markdown("---")
else:
    st.info("⚠️ Kategorik sütun bulunamadı.")



# ==============================================================
# 🔗 KORELASYON ANALİZİ
# ==============================================================
st.subheader("🔗 Korelasyon Matrisi (Sayısal Değişkenler)")
st.markdown("---")

if len(numeric_cols) >= 2:
    corr_method = st.selectbox("Korelasyon yöntemi seçin", ["pearson", "spearman", "kendall"])
    corr = df[numeric_cols].corr(method=corr_method)
    st.dataframe(corr, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
else:
    st.info("Korelasyon hesaplamak için en az iki sayısal sütun gereklidir.")


# ------------------- DAĞILIM ANALİZİ (4 TEST + RENK + Opsiyonel + Görselleştirme) -------------------
st.subheader("📊 Dağılım Analizi (Skew / Kurtosis + Normallik Testleri)")

if not numeric_cols:
    st.warning("⚠️ Sayısal sütun bulunamadı.")
else:
    # Kullanıcı sütun seçebilir veya None
    col_options = ["Yapmak istemiyorum"] + numeric_cols
    selected_col = st.selectbox("Analiz yapılacak sütunu seçin (opsiyonel)", col_options, key="dist_analysis_col")
    
    if selected_col == "Yapmak istemiyorum":
        st.info("Normallik testleri yapılmadı.")
    else:
        data = df[selected_col].dropna()
        n = len(data)

        # Hesaplamalar
        skew_val = float(stats.skew(data))
        kurt_val = float(stats.kurtosis(data, fisher=False))
        mean_val = float(np.mean(data))
        std_val = float(np.std(data, ddof=0))

        # Çarpıklık ve Basıklık yorumları
        skew_comment = ("Dağılım büyük ölçüde simetrik." if abs(skew_val)<0.5 
                        else "Sağa çarpık (pozitif skew)" if skew_val>=0.5
                        else "Sola çarpık (negatif skew)")
        kurt_comment = ("Basıklık ~3 (normal benzeri)" if 2.5<=kurt_val<=3.5 
                        else "Leptokurtik (tepe yüksek, kuyruklar kalın)" if kurt_val>3.5
                        else "Platykurtik (tepe basık, kuyruklar ince)")

        # --- Basıklık, Çarpıklık ve Aykırı Değer Kartları ---
        col1, col2, col3 = st.columns(3)

        # Çarpıklık kartı
        col1.markdown(f"""
        <div style="background:#F0F8FF;padding:10px;border-radius:8px;text-align:center;margin-bottom:10px">
            <div style="font-size:14px;color:#333"><b>📐 Çarpıklık (Skew)</b></div>
            <div style="font-size:22px;font-weight:700;color:black">{skew_val:.3f}</div>
            <div style="font-size:12px;color:#555;margin-top:6px;">{skew_comment}</div>
        </div>
        """, unsafe_allow_html=True)

        # Basıklık kartı
        col2.markdown(f"""
        <div style="background:#FFF8E1;padding:10px;border-radius:8px;text-align:center;margin-bottom:10px">
            <div style="font-size:14px;color:#333"><b>📏 Basıklık (Kurtosis)</b></div>
            <div style="font-size:22px;font-weight:700;color:black">{kurt_val:.3f}</div>
            <div style="font-size:12px;color:#555;margin-top:6px;">{kurt_comment}</div>
        </div>
        """, unsafe_allow_html=True)

        # Aykırı değer kartı (alt ve üst sınır yan yana)
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = (Q1 - (1.5 * IQR))
        upper_bound = (Q3 + (1.5 * IQR))
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / n) * 100 if n > 0 else 0

        # Kompakt aykırı değer kartı
        col3.markdown(f"""
        <div style="background:#FFE0B2;padding:6px;border-radius:6px;text-align:center;margin-bottom:6px">
            <div style="font-size:12px;color:#333"><b>📌 Aykırı Değerler</b></div>
            <div style="font-size:18px;font-weight:700;color:black">{outlier_count} / {n}</div>
            <div style="font-size:11px;color:#555;margin-top:2px;">%{outlier_pct:.2f} veri</div>
            <div style="display:flex; justify-content:space-around; margin-top:4px;">
                <div style="font-size:11px;color:#555;"><b>Alt sınır:</b> {lower_bound:.3f}</div>
                <div style="font-size:11px;color:#555;"><b>Üst sınır:</b> {upper_bound:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        # --- Normallik testleri ---
        # Veri küçükse öneri: Shapiro, orta: D’Agostino, büyük: KS
        if n < 50:
            recommendation_note = "Küçük veri seti, Shapiro-Wilk testi önerilir."
        elif n >= 50 :
            recommendation_note = "Orta boy veri seti, Kolmogorov-Smirnov testi önerilir."

        # Normallik testleri hesaplama
        with st.spinner("Testler çalıştırılıyor..."):
            try: sh_stat, sh_p = stats.shapiro(data)
            except: sh_stat, sh_p = (np.nan, np.nan)
            try: dp_stat, dp_p = stats.normaltest(data)
            except: dp_stat, dp_p = (np.nan, np.nan)
            try: ks_stat, ks_p = stats.kstest((data-mean_val)/std_val, 'norm')
            except: ks_stat, ks_p = (np.nan, np.nan)
            try:
                ad_result = stats.anderson(data, dist='norm')
                ad_stat = ad_result.statistic
                ad_comment = "Normal" if ad_stat < ad_result.critical_values[2] else "Anormal"
            except: ad_stat, ad_comment = (np.nan, "Hata")

        # --- Normallik Test Kartları ---
        tests = {
            "Shapiro-Wilk": {"stat": sh_stat, "p": sh_p},
            "D’Agostino-Pearson": {"stat": dp_stat, "p": dp_p},
            "Kolmogorov-Smirnov": {"stat": ks_stat, "p": ks_p},
            "Anderson-Darling": {"stat": ad_stat, "p": np.nan, "Yorum": ad_comment}
        }

        test_cols = st.columns(4)
        for i, (test_name, test_info) in enumerate(tests.items()):
            if test_name != "Anderson-Darling":
                p_val = test_info["p"]
                is_normal = False if np.isnan(p_val) else (p_val>0.05)
                bg_color = "#E8F6EC" if is_normal else "#FFEFEF"
                text_color = "#2e7d32" if is_normal else "#b71c1c"
                p_text = f"p = {p_val:.4f}" if not np.isnan(p_val) else "NaN"
            else:
                is_normal = True if test_info["Yorum"]=="Normal" else False
                bg_color = "#E8F6EC" if is_normal else "#FFEFEF"
                text_color = "#2e7d32" if is_normal else "#b71c1c"
                p_text = f"Stat = {test_info['stat']:.4f}" if not np.isnan(test_info['stat']) else "NaN"

            test_cols[i].markdown(f"""
            <div style="background:{bg_color};padding:10px;border-radius:8px;text-align:center;margin-right:5px;margin-left:5px;margin-bottom:15px">
                <div style="font-size:14px;color:{text_color};font-weight:700">{test_name}</div>
                <div style="font-size:18px;font-weight:700;color:black">{p_text}</div>
                <div style="font-size:12px;color:#555;margin-top:4px;">{"Normal dağılıma uygun" if is_normal else "Normal dağılımdan sapma"}</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(recommendation_note)

        full_data = df[selected_col]  # orijinal veri, eksikleri içeriyor
        data = full_data.dropna()     # analiz için temiz veri
        missing_count = full_data.isna().sum()
        if missing_count == 0:
            st.info("Veride eksik değer bulunmamaktadır, doldurma gerekmez.")
        else:
            st.warning(f"Veride {missing_count} eksik değer bulunmaktadır.")

        # --- Görselleştirmeler: Histogram + Q-Q ---
        st.markdown("---")
        st.markdown("### 📊 Dağılım Görselleştirmesi")
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(figsize=(10,5))
            sns.histplot(data, kde=True, ax=ax1, color="#4C72B0")
            ax1.axvline(mean_val, color='black', linestyle='--', linewidth=1)
            ax1.set_title(f"{selected_col} Histogram + KDE")
            st.pyplot(fig1)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(10,5.3))
            stats.probplot(data, dist="norm", plot=ax2)
            ax2.set_title(f"{selected_col} Q-Q Plot")
            st.pyplot(fig2)





