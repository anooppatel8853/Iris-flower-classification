import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="Iris Classifier & Visualizer", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classification App")
st.write("Yeh app Iris flower ke species ko predict karti hai aur alag-alag ML models aur data distribution ko visualize karti hai.")

# =========================================================
# SECTION 1: ALL 5 GRAPHS SECTION (Using Tabs)
# =========================================================
st.markdown("---")
st.subheader("📊 Data Visualization & Model Comparison")

try:
    # Load Dataset
    data = pd.read_csv("iris.csv")

    # Features & Target
    X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = data["Species"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5 alag-alag Tabs banana graphs ke liye
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Model Comparison", 
        "🍀 Sepal Analysis", 
        "🌺 Petal Analysis", 
        "⛓️ Feature Correlation", 
        "🧬 Pairplot Analysis"
    ])

    # --- TAB 1: Model Accuracy Bar Graph ---
    with tab1:
        st.write("### Model Accuracy Comparison")
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=3)
        }
        names = []
        accuracies = []
        for name, model_item in models.items():
            model_item.fit(X_train, y_train)
            y_pred = model_item.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            names.append(name)
            accuracies.append(acc)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        bars = ax1.bar(names, accuracies, color=['#4CAF50', '#2196F3', '#FF9800'])
        ax1.set_ylim(0.9, 1.05)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, height + 0.005, f"{height:.2f}", ha='center', va='bottom')
        st.pyplot(fig1)

    # --- TAB 2: Sepal Length vs Width (Scatter Plot) ---
    with tab2:
        st.write("### Sepal Length vs Sepal Width")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x="SepalLengthCm", y="SepalWidthCm", hue="Species", palette="Set1", ax=ax2)
        ax2.set_title("Sepal Dimensions")
        st.pyplot(fig2)

    # --- TAB 3: Petal Length vs Width (Scatter Plot) ---
    with tab3:
        st.write("### Petal Length vs Petal Width")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x="PetalLengthCm", y="PetalWidthCm", hue="Species", palette="Set2", ax=ax3)
        ax3.set_title("Petal Dimensions")
        st.pyplot(fig3)

    # --- TAB 4: Heatmap (Correlation) ---
    with tab4:
        st.write("### Feature Correlation Heatmap")
        # Sirf numeric columns ka correlation nikalne ke liye
        numeric_data = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
        st.pyplot(fig4)

    # --- TAB 5: Dataset Pairplot (🧬) ---
    with tab5:
        st.write("### Pairplot Analysis (All Features)")
        # Pairplot ke liye pairplot object ko direct pass karte hain
        fig5 = sns.pairplot(data, hue="Species", palette="husl")
        st.pyplot(fig5)

except FileNotFoundError:
    st.warning("⚠️ 'iris.csv' file nahi mili! Graphs dekhne ke liye 'iris.csv' ko GitHub par push karein.")
except Exception as e:
    st.error(f"Graph load karne mein koi issue aaya: {e}")


# =========================================================
# SECTION 2: LIVE PREDICTION
# =========================================================
st.markdown("---")
st.subheader("🔮 Make a Live Prediction")
st.write("Apne flower ke measurements enter karein:")

try:
    # Load Saved Model
    with open("model.pkl", "rb") as file:
        saved_model = pickle.load(file)

    col_input1, col_input2 = st.columns(2)
    with col_input1:
        sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1, format="%.1f", key="sl")
        sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5, format="%.1f", key="sw")
    with col_input2:
        petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4, format="%.1f", key="pl")
        petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, format="%.1f", key="pw")

    if st.button("Predict Flower Species", type="primary"):
        input_data = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        )

        prediction = saved_model.predict(input_data)
        st.balloons()
        st.success(f"🎉 Predicted Flower Species: **{prediction[0]}**")

except FileNotFoundError:
    st.error("❌ 'model.pkl' file nahi mili!")