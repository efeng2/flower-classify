import streamlit as st
import joblib
import numpy as np

# 加载模型
model = joblib.load('iris_model.joblib')

# 页面标题
st.title("Iris Flower Classification Dashboard")

# 输入字段
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=5.0, value=3.5, step=0.1)

# 预测按钮
if st.button("Predict"):
    # 使用模型进行预测
    features = np.array([[petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    st.success(f"Predicted Species: {species[prediction]}")