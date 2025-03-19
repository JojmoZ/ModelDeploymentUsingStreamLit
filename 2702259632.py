import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Iris 2702259632",
    layout="wide"
)


@st.cache_resource
def load_models():
    with open('2702259632_random.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    return rf_model

@st.cache_data
def load_data():
    return pd.read_csv('IRIS.csv')


try:
    rf_model = load_models()
    df = load_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    models_loaded = False


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Model Overview", "🔮 Predict"])




if page == "📊 Model Overview":
    st.title("📊 Model Overview")

    
    st.subheader("Iris Dataset")
    st.write(df.head())

    
    if models_loaded:
        st.subheader("Feature Importance")
        feature_importance = rf_model.feature_importances_
        features = df.columns[:-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feature_importance, y=features, palette="viridis", ax=ax)
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance in Random Forest")

        st.pyplot(fig)

    else:
        st.warning("Model not loaded, unable to display feature importance.")




elif page == "🔮 Predict":
    st.title("🔮 Make a Prediction")
    
    if models_loaded:
        
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

        
        if st.button("Predict"):
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = rf_model.predict(input_data)[0]
            
            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            predicted_species = species_map.get(prediction, "Unknown")

            st.success(f"🌸 Predicted Species: **{predicted_species}**")
    else:
        st.warning("Model not loaded. Please check the model file.")

