from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
def random_forest():
    df = pd.read_csv('train.csv')
    # 查看每列的缺失值数量及比例
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100

    print("缺失值统计：")
    print(pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage}))
    # 对少量缺失的数据列进行填充
    df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].median(), inplace=True)

    # 对中等缺失的数据列进行填充并创建缺失指示变量
    for col in ['Laptop Compartment', 'Waterproof', 'Style', 'Material', 'Size', 'Brand']:
        df[col + '_missing'] = df[col].isnull().astype(int)
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 处理大量缺失的数据列
    df.drop(columns=['Color'], inplace=True)  # 或者使用上面提到的方法之一
    # 或者
    # df['has_color'] = df['Color'].notnull().astype(int)
    # df.drop(columns=['Color'], inplace=True)

    # 检查是否还有缺失值
    print("处理后的缺失值统计：")
    print(df.isnull().sum())
    df = pd.get_dummies(df, drop_first=True)
    # 假设目标变量是'Price'
    X = df.drop('Price', axis=1)  # 特征
    y = df['Price']  # 目标变量
    # 假设X和y已经准备好
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化模型
    rf = RandomForestRegressor(random_state=42)

    # 定义参数网格
    param_grid = {
        'n_estimators': [20, 40],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 输出最佳参数
    print("Best parameters found: ", grid_search.best_params_)

    # 使用最佳参数重新训练模型
    best_rf = grid_search.best_estimator_

    # 预测
    y_pred = best_rf.predict(X_test)

    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def neural_network():
    # 加载数据
    train = pd.read_csv('train.csv', index_col=0)
    test = pd.read_csv('test.csv', index_col=0)

    # 处理类别型特征
    cat_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

    for col in cat_cols:
        train[col] = train[col].fillna('Unknown')
        test[col] = test[col].fillna('Unknown')

    # 定义交叉验证策略
    skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    scores = []
    test_preds = []
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    print(pd.isnull(train).sum())
    print(pd.isnull(test).sum())

    for i, (train_index, test_index) in enumerate(skf.split(train)):
        print(f"------------ Working on Fold {i} ------------")

        # 分割数据集
        X_train_fold, X_test_fold = train.iloc[train_index], train.iloc[test_index]
        y_train_fold, y_test_fold = X_train_fold.pop('Price'), X_test_fold.pop('Price')

        # 对类别型特征进行one-hot编码
        X_train_encoded = pd.get_dummies(X_train_fold, columns=cat_cols, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test_fold, columns=cat_cols, drop_first=True)
        X_train_encoded = X_train_encoded.astype(float)
        X_test_encoded = X_test_encoded.astype(float)
        # 确保测试集中的列与训练集匹配（可能有缺失的列）
        for col in X_train_encoded.columns:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0

        # 转换为Tensor
        X_train_tensor = torch.tensor(X_train_encoded.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_encoded.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_fold.values, dtype=torch.float32).view(-1, 1)

        # 构建神经网络模型
        model = Net(X_train_tensor.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        for epoch in range(10):  # epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            val_loss = criterion(outputs, y_test_tensor)
            print(f'Fold: {i}, MSE: {val_loss.item()}')
            predictions = outputs.numpy().flatten()
            score = mean_squared_error(y_test_fold, predictions, squared=False)
            print(f'Fold: {i}, RMSE: {score}')
            scores.append(score)
            test_predictions = model(torch.tensor(pd.get_dummies(test, columns=cat_cols, drop_first=True).values,
                                                  dtype=torch.float32)).numpy().flatten()
            test_preds.append(test_predictions)

    # 输出平均RMSE和标准差
    average_rmse = np.mean(scores)
    std_rmse = np.std(scores)
    print(f"The 5-fold average oof RMSE score of the Neural Network model is {average_rmse}")
    print(f"The 5-fold std oof RMSE score of the Neural Network model is {std_rmse}")




def neural_network2():
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

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
        test[col] = test[col].fillna('Unknown')

    # 定义交叉验证策略
    skf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    scores = []
    test_preds = []
    for i, (train_index, test_index) in enumerate(skf.split(train)):
        print(f"------------ Working on Fold {i} ------------")

        # 分割数据集
        X_train_fold, X_test_fold = train.iloc[train_index], train.iloc[test_index]
        y_train_fold, y_test_fold = X_train_fold.pop('Price'), X_test_fold.pop('Price')

        # 对类别型特征进行one-hot编码
        X_train_encoded = pd.get_dummies(X_train_fold, columns=cat_cols, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test_fold, columns=cat_cols, drop_first=True)
        X_val_encoded = pd.get_dummies(test, columns=cat_cols, drop_first=True)

        # 确保测试集中的列与训练集匹配（可能有缺失的列）
        for col in X_train_encoded.columns:
            if col not in X_test_encoded.columns:
                X_test_encoded[col] = 0

        # 构建神经网络模型
        model = Sequential([
            Dense(64, input_dim=X_train_encoded.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)  # 输出层，用于回归任务，不使用激活函数
        ])

        # 编译模型
        model.compile(optimizer=Adam(), loss='mean_squared_error')

        # 训练模型
        model.fit(X_train_encoded, y_train_fold, epochs=2, batch_size=32)
        loss = model.evaluate(X_test_encoded, y_test_fold, verbose=0)
        print(f'Fold: {i}, MSE: {loss}')
        # 预测并计算RMSE
        y_pred = model.predict(X_test_encoded).flatten()
        score = mean_squared_error(y_test_fold, y_pred, squared=False)
        print(f'Fold: {i}, RMSE: {score}')
        scores.append(score)
        test_preds.append(model.predict(X_val_encoded).flatten())

    # 输出平均RMSE和标准差
    average_rmse = np.mean(scores)
    std_rmse = np.std(scores)
    print(f"The 5-fold average oof RMSE score of the Neural Network model is {average_rmse}")
    print(f"The 5-fold std oof RMSE score of the Neural Network model is {std_rmse}")


neural_network2()