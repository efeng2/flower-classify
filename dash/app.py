import dash
from dash import dcc, html, Input, Output
import joblib
import numpy as np

# 加载模型
model = joblib.load('iris_model.joblib')

# 初始化 Dash 应用
app = dash.Dash(__name__)

# 定义布局
app.layout = html.Div([
    html.H1("Iris Flower Classification Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Petal Length (cm):"),
        dcc.Input(id='petal-length', type='number', value=5.1, step=0.1),
    ], style={'marginBottom': 10}),
    
    html.Div([
        html.Label("Petal Width (cm):"),
        dcc.Input(id='petal-width', type='number', value=3.5, step=0.1),
    ], style={'marginBottom': 20}),
    
    html.Button('Predict', id='predict-button', n_clicks=0, style={'marginBottom': 20}),
    
    html.Div(id='prediction-output', style={'fontSize': 20, 'fontWeight': 'bold'}),
])

# 定义回调函数
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('petal-length', 'value'), Input('petal-width', 'value')]
)
def update_output(n_clicks, petal_length, petal_width):
    if n_clicks > 0:
        # 使用模型进行预测
        features = np.array([[petal_length, petal_width]])
        prediction = model.predict(features)[0]
        species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        return f"Predicted Species: {species[prediction]}"
    return ""

# 运行应用
if __name__ == '__main__':
    app.run(debug=True)