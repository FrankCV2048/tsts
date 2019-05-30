import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#数据处理
data=np.loadtxt(open('data.txt','rb'),delimiter=',')

data_x=data[:,:2] #提取前两列作为数据
data_y=data[:,2]  #提取最后一列作为label
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, train_size=20) #随机划分训练数据与测试数据

#训练
model = LinearRegression(fit_intercept=True) # fit_intercept=True 均值为零 ，方差为一
model.fit(x_train, y_train)
k_de = model.intercept_  # 常量
x1_x2 = model.coef_  # 回归系数

#测试
# y_test=np.reshape(y_test,[-1,1])
# test_data=np.concatenate((x_test,y_test),axis=1)
# test_data = test_data[test_data[:,0].argsort()]
# x_test=test_data[:,:2] #提取前两列作为数据
# y_test=test_data[:,2]



predict=model.predict(x_test)


test_data = np.concatenate((x_test,np.reshape(predict,[-1,1]),np.reshape(y_test,[-1,1])),axis=1)
test_data = test_data[test_data[:,0].argsort()]
x_test=test_data[:,:2] #提取前两列作为数据
predict=test_data[:,2]
y_test=test_data[:,3]


plt.xlabel('序号')
plt.ylabel('销量')
plt.plot(y_test,'red',label="测试数据",marker='o')
plt.plot(predict,'black',label="预测数据",marker='D')
plt.legend(loc='upper right')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.show()
