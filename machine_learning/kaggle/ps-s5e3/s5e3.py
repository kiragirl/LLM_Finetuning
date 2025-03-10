from scipy.stats import uniform, randint, loguniform

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 添加导入
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
features = test.columns.tolist()
features.remove('day')


def draw_boxplot():
    print(features)
    for feat_name in features:
        # 绘制箱线图
        plt.boxplot(train[feat_name], patch_artist=True, boxprops=dict(facecolor="lightblue"))

        # 添加标题和标签
        plt.title(feat_name)
        plt.ylabel("Values")

        # 显示图形
        plt.show()


# 分割数据集
X = train.drop(['rainfall', 'day'], axis=1)  # 特征
test = test.drop(['day'], axis=1)  # 特征
y = train['rainfall']  # 目标变量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def train_lightgbm(X_train, X_val, y_train, y_val):
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_val = le_target.transform(y_val)
    # 定义 LightGBM 模型参数
    params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'auc',  # 评价指标为 AUC
        'boosting_type': 'gbdt',  # 使用 GBDT 算法
        'num_iterations': 2000,  # 迭代次数
        'learning_rate': 0.01,  # 学习率
        'max_depth': -1,  # 不限制深度
        'num_leaves': 128,  # 直接控制叶子节点数
        'subsample': 0.8,  # 子采样比例
        'colsample_bytree': 0.8,  # 特征子采样比例
        'random_state': 42,  # 随机种子
        'verbose': -1  # 不输出日志
    }

    # 创建 Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),  # 启用早停
            lgb.log_evaluation(10)  # 每 10 次迭代输出一次日志
        ]
    )

    # 在验证集上预测
    y_pred = model.predict(X_val)
    print(y_pred)
    # 计算 AUC
    auc_score = roc_auc_score(y_val, y_pred)
    print(f"Validation AUC: {auc_score}")
    # 将概率转换为分类结果
    threshold = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_threshold = 0
    pre_accuracy = 0
    for i in threshold:
        y_pred_binary = (y_pred >= i).astype(int)
        # 计算准确率
        accuracy = accuracy_score(y_val, y_pred_binary)
        if accuracy > pre_accuracy:
            pre_accuracy = accuracy
            best_threshold = i
        print(f"If Threshold is:{i}, then Accuracy: {accuracy:.8f}")
    print(f"Best Threshold is:{best_threshold}")


def predict_eval(best_model, test_pool, y_test):
    # 获取概率预测
    y_pred_prob = best_model.predict_proba(test_pool)[:, 1]
    # 获取类别预测
    y_pred = best_model.predict(test_pool)
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    # 输出分类报告
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def train_cat_boost(X_train, X_val, y_train, y_val):
    # 3. 初始化模型
    model = CatBoostClassifier(early_stopping_rounds=50)
    # 2. 定义参数分布
    # 定义参数分布
    param_dist = {
        'iterations': randint(200, 1000),
        'learning_rate': [0.1, 0.01, 0.001],
        'depth': [4, 6, 8],
        'l2_leaf_reg': randint(1, 5),
        'subsample': [0.5, 0.8],
        'colsample_bylevel': uniform(0.5, 1),
        'random_strength': loguniform(1e-8, 1),
        'loss_function': ['Logloss'],
        'eval_metric': ['AUC'],
        'verbose': [50]
    }

    # 设置 RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # 随机采样次数
        cv=5,  # 3 折交叉验证
        scoring='roc_auc',  # 评分标准
        verbose=1,  # 输出日志
        random_state=42,  # 随机种子
        n_jobs=-1  # 使用所有 CPU 核心
    )
    random_search.fit(X_train, y_train, verbose=50)
    # 提取最佳参数
    # best_params = grid_search.best_params_  # 如果是 GridSearchCV
    best_params = random_search.best_params_  # 如果是 RandomizedSearchCV

    print("Best Parameters:", best_params)
    # 使用最佳参数初始化模型
    model = CatBoostClassifier(**best_params)

    # 创建完整的数据池
    # full_pool = Pool(data=X, label=y, cat_features=categorical_features)
    # 初始化模型
    # model = CatBoostClassifier(
    #     loss_function='Logloss',  # 设置损失函数为 Logloss
    #     eval_metric='AUC',  # 设置评价指标为 AUC
    #     iterations=1000,  # 减少迭代次数
    #     learning_rate=0.05,  # 提高学习率
    #     depth=6,  # 降低树深度
    #     subsample=0.8,  # 降低子采样比例
    #     colsample_bylevel=0.8,  # 降低特征子采样比例
    #     random_strength=1,  # 减少随机性
    #     # task_type='GPU',  # 启用 GPU 加速
    #     early_stopping_rounds=50,  # 启用早期停止
    #     verbose=10  # 每 10 次迭代输出一次日志
    # )
    train_pool = Pool(data=X_train, label=y_train)
    test_pool = Pool(data=X_val, label=y_val)
    # 训练模型
    model.fit(train_pool, eval_set=test_pool)
    # 在完整数据集上训练模型
    # best_model.fit(train_pool, verbose=200)  # 设置 verbose 控制日志输出频率
    # 保存模型
    model.save_model('catboost_best_model.cbm')

    # （可选）如果需要保存为其他格式，例如 JSON
    # model.save_model('catboost_best_model.json', format='json')
    # 5. 输出最佳参数和结果
    # print("Best Parameters:", random_search.best_params_)
    # print("Best Score:", random_search.best_score_)
    # 输出模型在验证集上的 AUC 值
    auc_score = model.best_score_['validation']['AUC']
    print(f"Validation AUC: {auc_score}")
    predict_eval(model, test_pool, y_val)


if __name__ == '__main__':
    # train_lightgbm(X_train, X_val, y_train, y_val)
    train_cat_boost(X_train, X_val, y_train, y_val)
