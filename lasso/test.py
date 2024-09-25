import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt

data = pd.read_csv("train_2005to2019.csv", parse_dates=["Date"])
data["Date"] = pd.to_datetime(data["Date"])


def company_data(company, start_time, end_time):
    column_list = ["Date", company]
    specific_data = data[column_list]
    return specific_data[
        (specific_data["Date"] > start_time) & (specific_data["Date"] < end_time)
    ]


def data_processing(data, seq_length):
    X_train, y_train = [], []
    for i in range(seq_length, len(data)):
        X_train.append(data[i - seq_length : i, 0])
        y_train.append(data[i, 0])
    return np.array(X_train), np.array(y_train)


# 训练
company = "AA"
start_time = dt.datetime(2005, 1, 3)
end_time = dt.datetime(2019, 9, 30)
company_stock_price = company_data(company, start_time, end_time)

# 数据预处理
price = np.array(company_stock_price[company])
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(price.reshape(-1, 1))  # 归一化
dev_size = 0.2  # 划分训练集和验证集
split = int(len(normalized_data) * (1 - dev_size))
train_data, dev_data = normalized_data[:split], normalized_data[split:]
PAST_LENGTH = 60
X_train, y_train = data_processing(train_data, PAST_LENGTH)
X_dev, y_dev = data_processing(dev_data, PAST_LENGTH)

# 选择最佳的 alpha 参数
# 定义评估标准（均方误差）
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
# 设定参数网格（先用 param_grid1 筛选出 alpha 的数量级，再用 param_grid2 筛选出最优值）
param_grid1 = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
param_grid2 = {
    "alpha": [
        0.00002,
        0.00003,
        0.00004,
        0.00005,
        0.00006,
        0.00007,
        0.00008,
        0.00009,
        0.0001,
        0.0002,
        0.0003,
        0.0004,
        0.0005,
        0.0006,
        0.0007,
        0.0008,
        0.0009,
    ]
}
# 初始化 GridSearchCV：指定 Lasso 模型，参数范围，交叉验证和评分标准
grid = GridSearchCV(Lasso(), param_grid2, cv=5, scoring=mse_scorer)
# 执行网格搜索
grid.fit(X_train, y_train)
# 打印结果
best_alpha = grid.best_params_["alpha"]
print(f"最佳的 alpha 参数为：{best_alpha}")
print("每次验证的均方误差 (MSE) 结果：")
means = grid.cv_results_["mean_test_score"]  # 每次交叉验证的得分（均方误差）
stds = grid.cv_results_["std_test_score"]  # 每次交叉验证得分的标准差
for mean, std, alpha in zip(means, stds, param_grid2["alpha"]):
    print(f"Alpha: {alpha}, MSE (mean): {-mean:.10f}, Std: {std:.7f}")

# Lasso回归
lasso_model = Lasso(alpha=best_alpha)  # alpha：正则化参数
lasso_model.fit(X_train, y_train)

# 预测
train_predict = lasso_model.predict(X_train)
dev_predict = lasso_model.predict(X_dev)

# 逆归一化
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
dev_predict = scaler.inverse_transform(dev_predict.reshape(-1, 1))

# 可视化
look_back = PAST_LENGTH

trainPredictPlot = np.empty_like(price)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back : len(train_predict) + look_back] = train_predict.flatten()

devPredictPlot = np.empty_like(price)
devPredictPlot[:] = np.nan
test_start = len(price) - len(dev_predict)
devPredictPlot[test_start:] = dev_predict.flatten()

original_scaled_data = scaler.inverse_transform(normalized_data)

plt.figure(figsize=(15, 6))
plt.plot(original_scaled_data, color="black", label=f"Actual {company} price")
plt.plot(trainPredictPlot, color="red", label=f"Predicted {company} price (train set)")
plt.plot(devPredictPlot, color="blue", label=f"Predicted {company} price (dev set)")

plt.title(f"{company} price prediction")
plt.xlabel("time")
plt.ylabel(f"{company} closing price")
plt.legend()
plt.show()


# 测试
data = pd.read_csv("test_2019to2020.csv", parse_dates=["Date"])
data["Date"] = pd.to_datetime(data["Date"])

company = "AA"
start_time = dt.datetime(2019, 7, 7)
end_time = dt.datetime(2020, 4, 2)
company_stock_price = company_data(company, start_time, end_time)

# 数据预处理
price = np.array(company_stock_price[company])
normalized_data = scaler.transform(price.reshape(-1, 1))
X_test, y_test = data_processing(normalized_data, PAST_LENGTH)

# 预测
test_predict = lasso_model.predict(X_test)

# 逆归一化
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 可视化
testPredictPlot = np.empty_like(price)
testPredictPlot[:] = np.nan
testPredictPlot[PAST_LENGTH:] = test_predict.flatten()

original_scaled_data = scaler.inverse_transform(normalized_data)

plt.figure(figsize=(15, 6))
plt.plot(original_scaled_data, color="black", label=f"Actual {company} price")
plt.plot(testPredictPlot, color="blue", label=f"predicted {company} price(test set)")
plt.title(f"{company} price prediction")
plt.xlabel("time")
plt.ylabel(f"{company} closing price")
plt.legend()
plt.show()
