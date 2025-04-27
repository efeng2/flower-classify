# Flower Classification Dashboard

A classic machine learning project using the Iris dataset from `sklearn.datasets`, featuring an base interactive dashboard built with **Dash** and **Streamlit**.

---

## Dataset
The project uses the **Iris dataset** (`load_iris`) from `sklearn.datasets`. This dataset contains 150 samples of iris flowers, with four features (sepal length, sepal width, petal length, petal width) and three species labels (`setosa`, `versicolor`, `virginica`).

---

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flower-classify-dashboard.git
   cd flower-classify-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

### Using Dash
To run the Dash-based dashboard:
```bash
python app.py
```

### Using Streamlit
To run the Streamlit-based dashboard:
```bash
streamlit run app.py
```

---

## Features
- **Interactive Input**: Users can input petal length and petal width to predict the species of an iris flower.
- **Real-Time Predictions**: The dashboard uses a pre-trained logistic regression model to classify flowers in real time.
- **Two Dashboard Options**:
  - **Dash**: A highly customizable dashboard framework for complex interactions.
  - **Streamlit**: A simple and intuitive framework for quick prototyping.

---

## License
This project is licensed under the [MIT License](https://mit-license.org/). Feel free to use, modify, and distribute it as needed.