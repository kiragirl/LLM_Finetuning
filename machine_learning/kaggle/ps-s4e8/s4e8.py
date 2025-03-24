import numpy as np  # linear algebra
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, log_evaluation, early_stopping
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import LabelEncoder as LEncoder
from sklearn.preprocessing import LabelEncoder


# class LabelEncoder(LEncoder):
#     def fit_transform(self, y):
#         """
#         在训练集中引入 'Unknown' 类别
#         """
#         return super(LabelEncoder, self).fit_transform(list(y) + ['Unknown'])
#
#     def transform(self, y):
#         """
#         将新标签标记为 'Unknown'
#         """
#         new_y = ['Unknown' if x not in set(self.classes_) else x for x in y]
#         return super(LabelEncoder, self).transform(new_y)


def train_lightgbm_random_search():
    train = pd.read_csv('train.csv', index_col=0)
    print('The dimension of the train dataset is:', train.shape)
    X = train.drop(columns=['class'])  # 特征列
    y = train['class']  # 目标列
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    le_target = LabelEncoder()
    le_target.fit(y_train)
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(categorical_features)
    # 选择数值列
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    print("数值列:", numeric_columns)
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    X[categorical_features] = X[categorical_features].fillna("missing")
    cat_feat_encoders = {}
    for col in categorical_features:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            cat_feat_encoders[col] = le
            X_train[col] = le.fit_transform(X_train[col])

    params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'auc',  # 评价指标为 AUC
        'boosting_type': 'gbdt',  # 使用 GBDT 算法
        # 'num_iterations': 1000,  # 迭代次数
        'learning_rate': 0.05,  # 学习率
        'max_depth': -1,  # 不限制深度
        'num_leaves': 256,  # 直接控制叶子节点数
        'subsample': 0.8,  # 子采样比例
        'colsample_bytree': 0.8,  # 特征子采样比例
        'random_state': 42,  # 随机种子
    }

    callbacks = [
        log_evaluation(period=10),  # 每 100 次迭代打印一次日志
        early_stopping(stopping_rounds=30)  # 如果验证集指标在 30 次迭代内未提升，则停止训练
    ]

    train_data = lgb.Dataset(X_train, label=y_train)
    # 使用 lgb.cv 进行交叉验证
    cv_results = lgb.cv(params, train_data, num_boost_round=1000, nfold=5, stratified=True, callbacks=callbacks)
    # print(cv_results)
    # 输出最优迭代次数和得分
    best_num_boost_round = len(cv_results['valid auc-mean'])
    print("Best Num Boost Round:", best_num_boost_round)
    print("Best AUC Score:", max(cv_results['valid auc-mean']))


def train_lightgbm():
    train = pd.read_csv('train.csv', index_col=0)
    print('The dimension of the train dataset is:', train.shape)
    X = train.drop(columns=['class'])  # 特征列
    y = train['class']  # 目标列
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(categorical_features)
    # 选择数值列
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    print("数值列:", numeric_columns)
    means = X[numeric_columns].mean()
    X_train[numeric_columns] = X_train[numeric_columns].fillna(means)
    X_test[numeric_columns] = X_test[numeric_columns].fillna(means)
    X[categorical_features] = X[categorical_features].fillna("missing")
    cat_feat_encoders = {}

    for col in categorical_features:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            cat_feat_encoders[col] = le
            X_train[col] = le.fit_transform(X_train[col])
    for col in categorical_features:
        if X_test[col].dtype == 'object':
            le = cat_feat_encoders[col]
            X_test[col] = [x if x in le.classes_ else -1 for x in X_test[col]]
            le.classes_ = np.append(le.classes_, -1)
            X_test[col] = le.transform(X_test[col])
    # 定义 LightGBM 模型参数
    params = {
        'objective': 'binary',  # 二分类任务
        'metric': 'auc',  # 评价指标为 AUC
        'boosting_type': 'gbdt',  # 使用 GBDT 算法
        'num_iterations': 2000,  # 迭代次数
        'learning_rate': 0.01,  # 学习率
        'max_depth': -1,  # 不限制深度
        'num_leaves': 512,  # 直接控制叶子节点数
        'subsample': 0.8,  # 子采样比例
        'colsample_bytree': 0.6,  # 特征子采样比例
        'random_state': 42,  # 随机种子
        'verbose': -1  # 不输出日志
    }

    # 创建 Dataset
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    valid_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features, reference=train_data)

    # 训练模型
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
    y_pred = model.predict(X_test)
    print(y_pred)
    # 计算 AUC
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Validation AUC: {auc_score}")
    # 将概率转换为分类结果
    threshold = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_threshold = 0
    pre_accuracy = 0
    for i in threshold:
        y_pred_binary = (y_pred >= i).astype(int)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred_binary)
        if accuracy > pre_accuracy:
            pre_accuracy = accuracy
            best_threshold = i
        print(f"If Threshold is:{i}, then Accuracy: {accuracy:.8f}")
    print(f"Best Threshold is:{best_threshold}")


def train_best_model():
    train = pd.read_csv('train.csv', index_col=0)
    print('The dimension of the train dataset is:', train.shape)
    X = train.drop(columns=['class'])  # 特征列
    y = train['class']  # 目标列

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(categorical_features)
    # 选择数值列
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    print("数值列:", numeric_columns)
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
    X[categorical_features] = X[categorical_features].fillna("missing")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. 初始化模型
    # model = CatBoostClassifier(early_stopping_rounds=50)
    # # 2. 定义参数分布
    # # 定义参数分布
    # param_dist = {
    #     'iterations': randint(200, 500),
    #     'learning_rate': [0.1, 0.01],
    #     'depth': [6],
    #     # 'l2_leaf_reg': randint(1, 5),
    #     'subsample': [0.5],
    #     'colsample_bylevel': uniform(0.5, 0.8),
    #     'random_strength': loguniform(1e-8, 1),
    #     'loss_function': ['Logloss'],
    #     'eval_metric': ['AUC'],
    #     'verbose': [1]
    # }
    #
    # # 设置 RandomizedSearchCV
    # random_search = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_dist,
    #     n_iter=10,  # 随机采样次数
    #     cv=2,  # 3 折交叉验证
    #     scoring='roc_auc',  # 评分标准
    #     verbose=1,  # 输出日志
    #     random_state=42,  # 随机种子
    #     n_jobs=-1  # 使用所有 CPU 核心
    # )
    # random_search.fit(X_train, y_train, cat_features=categorical_features, verbose=10)
    # # 提取最佳参数
    # # best_params = grid_search.best_params_  # 如果是 GridSearchCV
    # best_params = random_search.best_params_  # 如果是 RandomizedSearchCV
    #
    # print("Best Parameters:", best_params)
    # # 创建训练池和测试池
    # train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    # test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)
    # # 使用最佳参数初始化模型
    # best_model = CatBoostClassifier(**best_params)

    # 创建完整的数据池
    # full_pool = Pool(data=X, label=y, cat_features=categorical_features)
    # 初始化模型
    model = CatBoostClassifier(
        loss_function='Logloss',  # 设置损失函数为 Logloss
        eval_metric='AUC',  # 设置评价指标为 AUC
        iterations=1000,  # 减少迭代次数
        learning_rate=0.05,  # 提高学习率
        depth=6,  # 降低树深度
        subsample=0.8,  # 降低子采样比例
        colsample_bylevel=0.8,  # 降低特征子采样比例
        random_strength=1,  # 减少随机性
        # task_type='GPU',  # 启用 GPU 加速
        early_stopping_rounds=50,  # 启用早期停止
        verbose=10  # 每 10 次迭代输出一次日志
    )
    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)
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
    predict_eval(model, test_pool, y_test)


def test_best_model():
    test = pd.read_csv('test.csv', index_col=0)
    X_test = test  # 特征列
    categorical_features = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    print(categorical_features)
    # 选择数值列
    numeric_columns = X_test.select_dtypes(include=[np.number]).columns.tolist()
    print("数值列:", numeric_columns)
    train = pd.read_csv('train.csv', index_col=0)
    means = train[numeric_columns].mean()
    X_test[numeric_columns] = X_test[numeric_columns].fillna(means)
    X_test[categorical_features] = X_test[categorical_features].fillna("missing")
    test_pool = Pool(data=X_test, cat_features=categorical_features)
    best_model = CatBoostClassifier()
    best_model.load_model('catboost_best_model.cbm')
    predict(best_model, test_pool)


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


def predict(best_model, test_pool):
    # 获取概率预测
    y_pred_prob = best_model.predict_proba(test_pool)[:, 1]
    # 获取类别预测
    y_pred = best_model.predict(test_pool)
    # 计算评估指标
    print(y_pred_prob)
    print(y_pred)


if __name__ == "__main__":
    # train_best_model()
    # test_best_model()
    train_lightgbm()
    # train_lightgbm_random_search()
