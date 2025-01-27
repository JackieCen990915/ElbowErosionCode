import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np

try:
    # 1. 导入数据
    df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
except FileNotFoundError:
    print("指定的 Excel 文件未找到，请检查文件路径。")
    exit(1)

# 2. 提取特征变量
x = df.drop(columns='er')
y = df['er']

print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))

# 3. 划分数据集
x_train_primary, x_test, y_train_primary, y_test = train_test_split(x, y, test_size=0.2, random_state=90)
x_train, x_val, y_train, y_val = train_test_split(x_train_primary, y_train_primary, test_size=0.25, random_state=123)

# 4. 数据归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 5. 定义并训练基学习器
base_model_1 = MLPRegressor(
    alpha=9.577699892831528e-05,
    hidden_layer_sizes=13,
    activation='relu',
    solver='lbfgs',
    random_state=90)
base_model_2 = RandomForestRegressor(
    n_estimators=20,
    max_depth=6,
    max_features=4,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=90)

base_model_1.fit(x_train, y_train)
base_model_2.fit(x_train, y_train)

# 6. 基学习器预测
predictions_1_val = base_model_1.predict(x_val)
predictions_2_val = base_model_2.predict(x_val)

predictions_1_test = base_model_1.predict(x_test)
predictions_2_test = base_model_2.predict(x_test)

# 7. 定义并训练元学习器
meta_model = SVR(kernel='rbf',
                   C=14.063456167204542,
                   epsilon=3.363135491649024e-05,
                   gamma=10.421706507320557)
meta_model.fit(np.column_stack((predictions_1_val, predictions_2_val)), y_val)

# 8. 元学习器预测
y_pred = meta_model.predict(np.column_stack((predictions_1_test, predictions_2_test)))

# 9. 模型评估
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred)
R2 = metrics.r2_score(y_test, y_pred)
EV = metrics.explained_variance_score(y_test, y_pred)

print('测试集评估结果:')
print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)
print('EV:', EV)