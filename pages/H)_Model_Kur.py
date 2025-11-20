# pages/model_kur.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Kur", page_icon="🤖", layout="wide")
st.title("🤖 Model Kur & Değerlendir")
st.markdown("---")

# ✅ Veri kontrolü
if "uploaded_df" not in st.session_state:
    st.warning("Önce veri yükleyin!")
    st.stop()

df = st.session_state["uploaded_df"].copy()

# ------------------ Hedef & Feature ------------------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
target = st.selectbox("🎯 Tahmin Edilecek Değişken (Y) — (Sadece Sayısal)", [None] + numeric_cols)

if target is None:
    st.warning("Lütfen sayısal bir hedef değişken seçin.")
    st.stop()

feature_candidates = [c for c in df.columns if c != target]
features = st.multiselect("🔧 Bağımsız Değişkenler (X)", feature_candidates, default=feature_candidates)

if not features:
    st.warning("En az 1 bağımsız değişken seçmelisiniz.")
    st.stop()

# Dummy Encoding (drop_first=False → SHAP uyumu)
X = pd.get_dummies(df[features], drop_first=False)
y = df[target]

# Eksik veri kontrolü
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    st.error("⚠️ Eksik veri bulunuyor → Eksik Veri İşleme sayfasına geçin.")
    st.stop()

# ------------------ Zaman Serisi Kontrolü ------------------
date_like_cols = [c for c in df.columns if "tarih" in c.lower() or "date" in c.lower()]
is_time_series = len(date_like_cols) > 0

if is_time_series:
    st.info(f"⏱️ Zaman serisi veri tespit edildi → {date_like_cols}. Veri sırası bozulmadan bölünecek.")
    df = df.sort_values(by=date_like_cols[0])  # ilk tarih sütununa göre sırala

# Train-test
test_size = st.slider("Test oranı (%)", 5, 50, 20) / 100
if is_time_series:
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=42
    )


# ✅ Train - Test kolon yapısı eşitlensin
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# ------------------ Scale opsiyonel ------------------
normalize_data = st.checkbox("🔄 StandardScaler ile ölçekle", True)

def scale_data():
    if normalize_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ✅ Scaler parametrelerini kaydet
        st.session_state["scaler_mean_"]  = scaler.mean_
        st.session_state["scaler_scale_"] = scaler.scale_

        return X_train_scaled, X_test_scaled
    
    return X_train, X_test


# ------------------------------------------
# TEK MODEL EĞİT
# ------------------------------------------
st.subheader("🧠 Tek Model Eğitimi")

model_name = st.selectbox("Model Seçin", [
    "Linear Regression", "Ridge Regression", "Lasso Regression",
    "Decision Tree", "Random Forest", "LightGBM", "XGBoost"
])

if model_name == "Linear Regression": model = LinearRegression()
elif model_name == "Ridge Regression": model = Ridge(alpha=st.slider("Alpha", 0.01, 10.0, 1.0))
elif model_name == "Lasso Regression": model = Lasso(alpha=st.slider("Alpha", 0.01, 10.0, 1.0))
elif model_name == "Decision Tree":    model = DecisionTreeRegressor(max_depth=st.slider("Max Depth", 1, 30, 6), random_state=42)
elif model_name == "Random Forest":    model = RandomForestRegressor(n_estimators=st.slider("Ağaç Sayısı", 50, 500, 200, 50), max_depth=st.slider("Max Depth", 2, 30, 8), random_state=42)
elif model_name == "LightGBM":         model = LGBMRegressor(n_estimators=st.slider("Ağaç Sayısı", 50, 500, 200, 50), learning_rate=st.slider("Learning Rate", 0.01, 0.3, 0.1), random_state=42)
elif model_name == "XGBoost":          model = XGBRegressor(n_estimators=st.slider("Ağaç Sayısı", 50, 500, 200, 50), learning_rate=st.slider("Learning Rate", 0.01, 0.3, 0.1), random_state=42)

if st.button("🚀 Modeli Eğit"):
    
    # 🔄 Ölçekleme yapıldıysa scaler parametrelerini kaydet
    if normalize_data:
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)

        # ✅ Scaler değerlerini Session State'e kaydet
        st.session_state["scaler_mean_"] = scaler.mean_
        st.session_state["scaler_scale_"] = scaler.scale_
    else:
        X_train_use, X_test_use = X_train, X_test

    model.fit(X_train_use, y_train)
    preds = model.predict(X_test_use)

    # ✅ Tüm gerekli session_state’leri kaydet
    st.session_state["model"] = model
    st.session_state["test_predictions"] = preds
    st.session_state["test_truth"] = y_test
    st.session_state["X_columns"] = X_train.columns
    st.session_state["X_full"] = X
    st.session_state["X_test_full"] = X_test
    st.session_state["X_test_scaled"] = X_test_use
    st.session_state["X_train_scaled"] = X_train_use
    st.session_state["uploaded_df"] = df

    # ✅ EKLENENLER
    st.session_state["features"] = features
    st.session_state["target_name"] = target

    st.success(f"✅ Model Eğitildi → R²={r2_score(y_test, preds):.3f} | MAE={mean_absolute_error(y_test, preds):.3f} | RMSE={mean_squared_error(y_test, preds, squared=False):.3f}")


# ------------------------------------------
# MODEL KARŞILAŞTIR & TUNING
# ------------------------------------------
st.subheader("🧪 Modelleri Karşılaştır ve Optimize Et")
multi_compare = st.checkbox("Modelleri karşılaştırmayı etkinleştir", False)

if multi_compare:

    base_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    X_train_use, X_test_use = scale_data()

    results = []
    trained_models = {}

    for name, mdl in base_models.items():
        mdl.fit(X_train_use, y_train)
        p = mdl.predict(X_test_use)
        trained_models[name] = mdl
        results.append([name, r2_score(y_test, p), mean_absolute_error(y_test, p), mean_squared_error(y_test, p, squared=False)])

    compare_df = pd.DataFrame(results, columns=["Model","R²","MAE","RMSE"]).sort_values("R²", ascending=False)
    st.dataframe(compare_df, use_container_width=True)

    # ------------------ Model Kaydet ------------------
    st.markdown("### 📌 Kaydedilecek model:")
    to_save = st.selectbox("Model seç:", compare_df["Model"].tolist())

    if st.button("📌 Bu modeli rapora kaydet"):
        chosen = trained_models[to_save]
        preds_save = chosen.predict(X_test_use)

        st.session_state["model"] = chosen
        st.session_state["test_predictions"] = preds_save
        st.session_state["test_truth"] = y_test
        st.session_state["X_columns"] = X_train.columns
        st.session_state["X_full"] = X
        st.session_state["X_test_scaled"] = X_test_use
        st.session_state["X_train_scaled"] = X_train_use
        st.session_state["X_test_full"] = X_test
        st.session_state["uploaded_df"] = df
        st.session_state["features"] = features
        st.session_state["target_name"] = target

        st.success("✅ Model kaydedildi → Model Raporu sayfasına geçebilirsiniz.")
        st.rerun()

    # ------------------ GridSearch Tuning ------------------
    st.markdown("---")
    st.markdown("### ⚡ GridSearch ile Optimize Et")

    grid_params = {
        "Ridge Regression": {"alpha": [0.1, 1, 5, 10]},
        "Lasso Regression": {"alpha": [0.1, 1, 5, 10]},
        "Decision Tree": {"max_depth": [3, 5, 8, 12]},
        "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [5, 8, 12]},
        "LightGBM": {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2]},
        "XGBoost": {"n_estimators": [100, 200, 300], "learning_rate": [0.05, 0.1, 0.2]}
    }

    tune_choice = st.selectbox("Optimize edilecek model", [m for m in compare_df["Model"].tolist() if m in grid_params])

    if st.button("🔍 En iyi parametreleri bul"):
        with st.spinner("⏳ GridSearch çalışıyor..."):
            grid = GridSearchCV(base_models[tune_choice], grid_params[tune_choice], cv=3, scoring='r2', n_jobs=-1)
            grid.fit(X_train_use, y_train)

        tuned = grid.best_estimator_
        preds_tuned = tuned.predict(X_test_use)

        st.success(f"✅ Optimize Edilmiş → R²={r2_score(y_test, preds_tuned):.3f} | MAE={mean_absolute_error(y_test, preds_tuned):.3f} | RMSE={mean_squared_error(y_test, preds_tuned, squared=False):.3f}")
        st.info(f"🔧 En iyi parametreler: {grid.best_params_}")
        st.warning("💡 Bu değerleri yukarıdaki Tek Model Eğitim bölümüne girip **Tekrar Eğit** yaparak rapora kaydedebilirsiniz.")
