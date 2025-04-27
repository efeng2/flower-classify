from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

iris = load_iris()
# train test split
x = iris.data[:,2:4]
y = iris.target
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=1, test_size=50)

model = LogisticRegression()
model.fit(x_train, y_train)

ac = accuracy_score(y_test, model.predict(x_test))
print('Model Accuarcy: ', ac)

# plot
# N, M=500,500 # how many grid samples
# t1=np.linspace(0, 8, N)
# t2=np.linspace(0, 3, M)
# x1, x2=np.meshgrid(t1,t2)
# x_new=np.stack((x1.flat,x2.flat), axis=1)
# y_predict=model.predict(x_new)
# y_hat=y_predict.reshape(x1.shape)
# iris_cmap = ListedColormap(["#ACC6C0", "#FF8080", "#A0A0FF"])

# plt.pcolormesh(x1,x2,y_hat,cmap=iris_cmap)
# plt.scatter(x[y==0,0], x[y==0,1], s=30, c='g', marker='^')
# plt.scatter(x[y==1,0], x[y==1,1], s=30, c='r', marker='o')
# plt.scatter(x[y==2,0], x[y==2,1], s=30, c='b', marker='s')

# plt.rcParams['font.sans-serif']='Simhei'
# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')
# plt.show()

import joblib

# 保存模型
joblib.dump(model, 'iris_model.joblib')

# 加载模型
# loaded_model = joblib.load('iris_model.joblib')