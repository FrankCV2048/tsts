import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#数据处理
data=np.loadtxt(open('data.txt','rb'),delimiter=',')
data_x=data[:,:2] #提取前两列作为数据
data_y=data[:,2]  #提取最后一列作为label
X_train, X_test, Y_train, Y_test = train_test_split(data_x,data_y, train_size=20)
#训练
model = LinearRegression()
model.fit(X_train, Y_train)
k_de = model.intercept_  # 常量
x1_x2 = model.coef_  # 回归系数
print(k_de,x1_x2)

score = model.score(X_test, Y_test) #计算回归拟合程度
print(score)
#测试
Y_pred = model.predict(X_test)
print(Y_pred)