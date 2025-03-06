from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

print(train.info())
print("\n")
print(test.info())

# 处理类别型特征
cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

for col in cat_cols:
    train[col] = train[col].fillna('Unknown')
    train[col] = train[col].astype('category')

    test[col] = test[col].fillna('Unknown')
    test[col] = test[col].astype('category')

# 定义交叉验证策略
skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

scores = []
for i, (train_index, test_index) in enumerate(skf.split(train)):
    print(f"------------ Working on Fold {i} ------------")

    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    y_train, y_test = X_train.pop('Price'), X_test.pop('Price')

    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Fold: {i}, RMSE: {score}')
    scores.append(score)

# 输出平均RMSE和标准差
average_rmse = np.mean(scores)
std_rmse = np.std(scores)
print(f"The 5-fold average oof RMSE score of the RandomForestRegressor model is {average_rmse}")
print(f"The 5-fold std oof RMSE score of the RandomForestRegressor model is {std_rmse}")